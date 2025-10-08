# rnencodec_rtplayer.py
# Encodec Player Base (profile-agnostic)
# Extracted and refactored from V3_EncodecStreamRNN.ipynb


from typing import Optional, Sequence
import numpy as np

# for the rt synth
from realtime_synth.generators.base import BaseGenerator
from realtime_synth.utils import exp_map01
from realtime_synth_ui import build_synth_ui


import time

#required import for threaded version
from concurrent.futures import ThreadPoolExecutor
#####################################################################
#####################################################################


class EncodecRTPlayer(BaseGenerator):
    # normalized params in [0,1]
   

    # -----------------------
    def __init__(self, rnngen, sr, frame_rate, buffersize,   chunksize, hopsize,  init_norm_params=None, param_labels=None, warmupsteps=10):

        self.units_p=np.zeros_like(init_norm_params)
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
        # NOTE: assumes global sr and frame_rate are defined elsewhere in your code
        self.framesizesamples = sr // frame_rate  # e.g., 75; encoder is 75 fps

        self.currentchunkframe = 0  # nth frame in the chunk of audio we are playing
        self.seeding_len = self.chunksizeframes - self.framehopsize
        self.genaudioframe = 0      # mth frame we've generated in total

        self._last_error = None
        self._decodetime = 0.0
        self._callrecord = ""

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


    # -----------------------
    def generate(self, frames, sr):
        assert frames == self.buffersize, "ooh, you're in trouble if frames requested is different than the buffer size."

        # if self.amp <= 0.0 or self.freq <= 0.0:
        #     self._scratch.fill(0.0)
        #     return self._scratch

        # slice current hop
        endsamp = self.nextsample + self.buffersize
        y = self.thisaudioseq[self.nextsample:endsamp]
        self.nextsample = endsamp

        # NON-BLOCKING: if we just started a hop, see if the background result is ready
        if self.currentchunkframe == 0:
            self._try_collect_next()
            # if nothing in-flight, (re)start background worker
            if self._next_future is None:
                self._schedule_next_hop()

        # advance within hop; at hop boundary, try to swap
        self.currentchunkframe += 1
        if self.currentchunkframe == self.framehopsize:
            if self.nextaudioseq is not None:
                # swap in new hop (no copy; ensure float32)
                self.thisaudioseq = np.asarray(self.nextaudioseq, dtype=np.float32)
                self.nextaudioseq = None
                self._schedule_next_hop()  # immediately start computing the following hop
            else:
                # no hop ready → output SILENCE for one hop (your preference)
                msg = "missed hop swap"
                self._last_error = (self._last_error + " | " + msg) if self._last_error else msg
                self.thisaudioseq = np.zeros(self.framehopsize * self.framesizesamples, dtype=np.float32)
                # also (re)schedule next hop in case worker died
                if self._next_future is None:
                    self._schedule_next_hop()
            # reset for new hop window
            self.currentchunkframe = 0
            self.nextsample = 0

        # # Do post signal processing based on params if you need to 
        # # scale into scratch (avoids alloc every block)
        # np.multiply(y, self.amp, out=self._scratch, casting='unsafe')
        # return self._scratch

        #otherwise, just return the buffer
        return y

    # -----------------------
    # This first sets the norm_params, and the units_params which are just used for display (the norm_params are the ones sent to the synth)
    def set_params(self, norm_params):
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

        return [f"{label}: {val:.1f} label" for label, val in zip(self.param_labels, self.units_p)]
            
