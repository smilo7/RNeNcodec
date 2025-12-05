# rnencodec_rtplayer.py
# Encodec Player Base (profile-agnostic)
# Extracted and refactored from V3_EncodecStreamRNN.ipynb


from typing import Optional, Sequence
import numpy as np

import soxr 

# for the rt synth
from realtime_synth.generators.base import BaseGenerator
from realtime_synth.utils import exp_map01
from realtime_synth_ui import build_synth_ui


import time

#required import for threaded version
from concurrent.futures import ThreadPoolExecutor
#####################################################################
#####################################################################
class Up2x48kStream:
    """Feed N at 24 kHz → get exactly 2N at 48 kHz every call (pads during startup)."""
    def __init__(self, channels=1, dtype="float32", quality="HQ"):
        self.ch = int(channels)
        self.dtype = dtype
        self.rs = soxr.ResampleStream(24000, 48000, num_channels=self.ch, dtype=dtype, quality=quality)
        self.buf = np.zeros((0, self.ch), dtype=dtype) if self.ch > 1 else np.zeros(0, dtype=dtype)

    def process(self, y24):
        x = np.asarray(y24, dtype=self.dtype, order="C")
        if self.ch > 1 and x.ndim == 1:
            x = np.tile(x[:, None], (1, self.ch))
        y48_new = self.rs.resample_chunk(x)                         # stateful
        # append to queue
        self.buf = (np.concatenate([self.buf, y48_new], axis=0) if self.ch > 1
                    else np.concatenate([self.buf, y48_new], axis=0))
        want = (x.shape[0] * 2)
        # pad during initial latency so we always return exactly 2N
        if self.buf.shape[0] < want:
            deficit = want - self.buf.shape[0]
            pad = (np.zeros((deficit, self.ch), dtype=self.dtype) if self.ch > 1
                   else np.zeros(deficit, dtype=self.dtype))
            out = (np.concatenate([self.buf, pad], axis=0))
            self.buf = self.buf[0:0]
            return out
        out = self.buf[:want]
        self.buf = self.buf[want:]
        return out
        
    
class EncodecRTPlayer(BaseGenerator):
    # normalized params in [0,1]
   

    # -----------------------
    def __init__(self, rnngen, sr, frame_rate, buffersize,   chunksize, hopsize,  init_norm_params=None, param_labels=None, warmupsteps=10, desc_vals=None):

        self.units_p=np.zeros_like(init_norm_params)
        self.desc_vals=desc_vals
        
        self._last_error = ""
        self._decodetime = 0.0
        self._callrecord = ""
        self._setnormval = ""
        
        super().__init__(init_norm_params or [0.5, 0.6])  # defaults
        print(f'Initialize EncodecRTPlayer')
        self.set_params(self.norm_params)  # initialize semantic values
        

        self.param_labels = param_labels
        

        self.rnngen = rnngen
        self.cond_size = rnngen.cond_size

        self.chunksizeframes = chunksize   # decode this many frames each time
        self.framehopsize    = hopsize     # decode a new chunk every framehopsize
        self.nextendframe    = self.framehopsize

        self.buffersize = buffersize
        self.nextsample = 0

        self.internal_sr=sr
        self.counter=0
        self.up2x= Up2x48kStream() # StatefulUp2x(channels=1, taps=64)
        # NOTE: assumes global sr and frame_rate are defined elsewhere in your code
        self.framesizesamples = sr // frame_rate  # e.g., 75; encoder is 75 fps

        self.currentchunkframe = 0  # nth frame in the chunk of audio we are playing
        self.seeding_len = self.chunksizeframes - self.framehopsize
        self.genaudioframe = 0      # mth frame we've generated in total



        # small scratch buffer to avoid per-callback allocations (optional)
        self._scratch = np.empty(self.buffersize, dtype=np.float32)

        self.thisaudioseq=np.zeros(self.framehopsize * self.framesizesamples)
        
        self.nextaudioseq = None  # will be filled by background worker

        # single background worker + a future for the next hop
        self._hop_exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="HopGen")
        self._next_future = None

        # Kick off the first async hop immediately
        self._schedule_next_hop()

        #and why not do a little warm up in case the user forgets - helps avoid noise at the begiing of a render
        if warmupsteps > 0 :
            rnngen.warmup(init_norm_params, warmupsteps)

    # -----------------------
    def _schedule_next_hop(self):
        """Launch getNextAudioHop() in the background (non-blocking)."""
        if self._next_future is None:
            try:
                self._next_future = self._hop_exec.submit(self.getNextAudioHop)
            except Exception as e:
                self._last_error = f"scheduling error: {e!r}"
                self._next_future = None

    # -----------------------
    def _try_collect_next(self):
        """
        If the background hop has finished, collect it into self.nextaudioseq (non-blocking).
        """
        fut = self._next_future
        if fut is not None and fut.done():
            try:
                self.nextaudioseq = fut.result()
            except Exception as e:
                self._last_error = f"hop result error: {e!r}"
                self.nextaudioseq = None
            finally:
                self._next_future = None  # allow scheduling the following hop

    # -----------------------
    def close(self):
        """Optional: call when tearing down to stop the worker quickly."""
        try:
            if hasattr(self, "_hop_exec") and self._hop_exec:
                self._hop_exec.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    # -----------------------
    def getNextAudioHop(self):
        """
        Heavy work: RNN inference + EnCodec decode for one hop.
        Runs on the background thread.
        Returns a 1D numpy array of length framehopsize * framesizesamples (mono, float32 or convertible).
        """
        self.genaudioframe = self.genaudioframe + self.framehopsize
        self._callrecord = self._callrecord + f";(start: {self.genaudioframe}, end: {self.genaudioframe + self.chunksizeframes})"

        start_time = time.monotonic()
        nextseq = np.array(self.rnngen.getNextAudioHop(self.norm_params[:self.cond_size], hop=self.framehopsize))
        
        self._decodetime += (time.monotonic() - start_time)

        # take only the hopsize of audio that we need
        return nextseq[-self.framehopsize * self.framesizesamples:]


    def generate(self, nsamples, sr):
        try:
            self.coiunter = self.counter+1
            # Are we returning at 24k (native) or 48k (upsampled)?
            do_resample = (sr != self.internal_sr)
    
            # How many 24k samples do we need to slice for this call?
            need24 = (nsamples + 1) // 2 if do_resample else nsamples
    
            # Slice current hop at 24k
            endsamp = self.nextsample + need24
            y = self.thisaudioseq[self.nextsample:endsamp]
            self.nextsample = endsamp
    
            # NON-BLOCKING: if we just started a hop, see if the background result is ready
            if self.currentchunkframe == 0:
                self._try_collect_next()
                if self._next_future is None:
                    self._schedule_next_hop()
    
            # advance within hop; at hop boundary, try to swap in the new hop buffer
            self.currentchunkframe += 1
            if self.currentchunkframe == self.framehopsize:
                if self.nextaudioseq is not None:
                    self.thisaudioseq = np.asarray(self.nextaudioseq, dtype=np.float32, order="C")
                    self.nextaudioseq = None
                    self._schedule_next_hop()
                else:
                    # missed hop: output silence for the next window and reschedule
                    self._last_error = (" | ".join(filter(None, [self._last_error, "missed hop swap"])))
                    self.thisaudioseq = np.zeros(self.framehopsize * self.framesizesamples, dtype=np.float32)
                    if self._next_future is None:
                        self._schedule_next_hop()
                self.currentchunkframe = 0
                self.nextsample = 0
    
            if not do_resample:
                # Return exactly nsamples @ 24k (pad if we ran short for any reason)
                if y.shape[0] < nsamples:
                    y = np.pad(y, (0, nsamples - y.shape[0]))
                else:
                    y = y[:nsamples]
                return y.astype(np.float32, copy=False)
    
            # --- 24k -> 48k (stateful, exact length) ---
            y24_c = np.asarray(y, dtype=np.float32, order="C")
            y48 = self.up2x.process(y24_c)   # <-- use the instance you created in __init__
    
            # Guarantee exact nsamples at output rate
            if y48.shape[0] < nsamples:
                #self._last_error = self._last_error + " | " f"zero pad because {y48.shape[0]} is less than {nsamples}"
                y48 = np.pad(y48, (0, nsamples - y48.shape[0]))
            else:
                y48 = y48[:nsamples]
            #self._last_error = y48 # self._last_error + " | " f"call up2x.process count={self.counter} y24_c[0]={y24_c[0]}"
            return y48.astype(np.float32, copy=False)
    
        except Exception as e:
            # stash a readable error the UI can surface
            self._last_error = self._last_error + " | " f"generate() error: {e!r}"
            # fail-safe: return silence so the audio callback doesn't explode
            return np.zeros(nsamples, dtype=np.float32)

    
    # -----------------------
    # This first sets the norm_params, and the units_params which are just used for display (the norm_params are the ones sent to the synth)
    def set_params(self, norm_params):

        # self._setnormval += "; "
        for i in range(len(norm_params)) :
            if self.desc_vals != None:
                
                if self.desc_vals[i] != 0:
                    norm_params[i]=round(norm_params[i]*(self.desc_vals[i]-1))/(self.desc_vals[i]-1) # -1 to get n-1 intervals with n points inclusive of endpoints 
                    # self._setnormval += f"{norm_params[i]} "
                    
        
        super().set_params(norm_params)
        # Map [0,1] → semantic values (your mapping)
        # self.freq = float(exp_map01(self.norm_params[0], 20.0, 2000.0))  # exponential Hz
        # self.amp  = float(self.norm_params[1])                           # linear gain 0..1

        for i in range(len(self.norm_params)) :
            self.units_p[i] = self.norm_params[i]

            # #if you want to "change the units" or the labels, you could do something like:
            # self.units_p[1] = 64 + self.norm_params[1]*12

    
    # -----------------------
    def formatted_readouts(self):
        # Optional: pretty labels shown next to sliders

        return [f"{label}: {val:.2f} label" for label, val in zip(self.param_labels, self.units_p)]
            
