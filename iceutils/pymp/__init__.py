"""

Copy (witih slight modification) of pymp package by Christoph Lassner:
https://github.com/classner/pymp

The MIT License (MIT)

Copyright (c) 2016 Christoph Lassner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# pylint: disable=invalid-name, bad-continuation, len-as-condition
from __future__ import print_function

import functools as _functools
import logging as _logging
import multiprocessing as _multiprocessing
import warnings as _warnings                                                                   
import os as _os
import platform as _platform
import signal as _signal
import sys as _sys
import time as _time

try:
    import numpy as _np

    _NP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NP_AVAILABLE = False

_NUM_PROCS = _multiprocessing.Value("i", 1, lock=False)  # pylint: disable=no-member

from . import config as _config

import queue as _Queue  # pylint: disable=import-error, E0102

_LOGGER = _logging.getLogger(__name__)

def Manager():
    """
    Factory for creating a Manager instance.
    """
    return _multiprocessing.Manager()


# pylint: disable=too-few-public-methods, too-many-instance-attributes
class Parallel(object):

    """A parallel region."""

    _level = 0
    _global_master = None

    def __init__(
        self, num_threads=None, manager=None, if_=True
    ):  # pylint: disable=redefined-outer-name
        if _platform.system().startswith("Windows") or _platform.system().startswith(
            "CYGWIN"
        ):
            raise Exception(
                "Pymp relies on full 'fork' support by the operating system. "
                "You seem to be using Windows, which unfortanetly does not "
                "do this."
            )
        self._num_threads = num_threads
        self._enabled = if_
        if not self._enabled:
            self._num_threads = 1
        self._is_fork = False
        self._pids = []
        self._thread_num = 0
        self._lock = None
        self._print_lock = None
        if Parallel._global_master is None:
            Parallel._global_master = _os.getpid()
        # Set manager
        self._manager = manager or _multiprocessing.Manager()
        # Dynamic schedule management.
        self._dynamic_queue = self._manager.Queue()
        self._iter_queue = None
        self._thread_loop_ids = None
        self._queuelock = _multiprocessing.Lock()
        # Exception management.
        self._exception_queue = self._manager.Queue()
        self._exception_lock = _multiprocessing.Lock()
        # Allows for self-checks.
        self._entered = False
        self._disposed = False

    def __enter__(self):
        _LOGGER.debug(
            "Entering `Parallel` context (level %d). Forking...", Parallel._level
        )
        # pylint: disable=global-statement
        assert len(self._pids) == 0, "A `Parallel` object may only be used once!"
        self._lock = _multiprocessing.Lock()
        self._print_lock = _multiprocessing.Lock()
        # pylint: disable=protected-access
        if self._num_threads is None:
            assert (
                len(_config.num_threads) == 1
                or len(_config.num_threads) > Parallel._level
            ), (
                "The value of PYMP_NUM_THREADS/OMP_NUM_THREADS must be "
                "either a single positive number or a comma-separated "
                "list of number per nesting level."
            )
            if len(_config.num_threads) == 1:
                self._num_threads = _config.num_threads[0]
            else:
                self._num_threads = _config.num_threads[Parallel._level]
        if not _config.nested:
            assert Parallel._level == 0, "No nested parallel contexts allowed!"
        Parallel._level += 1
        self._iter_queue = self._manager.Queue(maxsize=self._num_threads - 1)
        # pylint: disable=protected-access
        with self._lock:
            # Make sure that max threads is not exceeded.
            if _config.thread_limit is not None:
                # pylint: disable=protected-access
                num_active = _NUM_PROCS.value
                self._num_threads = min(
                    self._num_threads, _config.thread_limit - num_active + 1
                )
            _NUM_PROCS.value += self._num_threads - 1
        self._thread_loop_ids = self._manager.list([-1] * self._num_threads)
        for thread_num in range(1, self._num_threads):
            pid = _os.fork()
            if pid == 0:  # pragma: no cover
                # Forked process.
                self._is_fork = True
                self._thread_num = thread_num
                break
            else:
                # pylint: disable=protected-access
                self._pids.append(pid)
        if not self._is_fork:
            _LOGGER.debug("Forked to processes: %s.", str(self._pids))
        self._entered = True
        return self

    def __exit__(self, exc_t, exc_val, exc_tb):
        _LOGGER.debug("Leaving parallel region (%d)...", _os.getpid())
        if exc_t is not None:
            with self._exception_lock:
                self._exception_queue.put((exc_t, exc_val, self._thread_num))
        if self._is_fork:  # pragma: no cover
            _LOGGER.debug("Process %d done. Shutting down.", _os.getpid())
            _os._exit(1)  # pylint: disable=protected-access
        for pid in self._pids:
            _LOGGER.debug("Waiting for process %d...", pid)
            _os.waitpid(pid, 0)
        # pylint: disable=protected-access
        with self._lock:
            _NUM_PROCS.value -= len(self._pids)
        Parallel._level -= 1
        self._disposed = True
        # Take care of exceptions if necessary.
        if self._enabled:
            while not self._exception_queue.empty():
                exc_t, exc_val, thread_num = self._exception_queue.get()
                _LOGGER.critical(
                    "An exception occured in thread %d: (%s, %s).",
                    thread_num,
                    exc_t,
                    exc_val,
                )
                raise exc_t(exc_val)
        else:
            if not self._exception_queue.empty():
                raise
        _LOGGER.debug("Parallel region left (%d).", _os.getpid())

    def _assert_active(self):
        """Assert that the parallel region is active."""
        assert (
            self._entered and not self._disposed
        ), "The parallel context of this object is not active!"

    @property
    def thread_num(self):
        """The worker index."""
        self._assert_active()
        return self._thread_num

    @property
    def num_threads(self):
        """The number of threads in this context."""
        self._assert_active()
        return self._num_threads

    @property
    def lock(self):
        """Get a convenient, context specific lock."""
        self._assert_active()
        return self._lock

    @classmethod
    def print(cls, *args, **kwargs):
        """Print synchronized."""
        # pylint: disable=protected-access
        with self._print_lock:
            print(*args, **kwargs)
            _sys.stdout.flush()

    def range(self, start, stop=None, step=1):
        """
        Get the correctly distributed parallel chunks.

        This corresponds to using the OpenMP 'static' schedule.
        """
        self._assert_active()
        if stop is None:
            start, stop = 0, start
        full_list = range(start, stop, step)
        per_worker = len(full_list) // self._num_threads
        rem = len(full_list) % self._num_threads
        schedule = [
            per_worker + 1 if thread_idx < rem else per_worker
            for thread_idx in range(self._num_threads)
        ]
        # pylint: disable=undefined-variable
        start_idx = _functools.reduce(
            lambda x, y: x + y, schedule[: self.thread_num], 0
        )
        end_idx = start_idx + schedule[self._thread_num]
        return full_list[start_idx:end_idx]

    def xrange(self, start, stop=None, step=1):
        """
        Get an iterator for this threads chunk of work.

        This corresponds to using the OpenMP 'dynamic' schedule.
        """
        self._assert_active()
        if stop is None:
            start, stop = 0, start
        with self._queuelock:
            pool_loop_reached = max(self._thread_loop_ids)
            # Get this loop id.
            self._thread_loop_ids[self._thread_num] += 1
            loop_id = self._thread_loop_ids[self._thread_num]
            if pool_loop_reached < loop_id:
                # No thread reached this loop yet. Set up the queue.
                for idx in range(start, stop, step):
                    self._dynamic_queue.put(idx)
            # Iterate.
            return _QueueIterator(self._dynamic_queue, loop_id, self)

    def iterate(self, iterable, element_timeout=None):
        """
        Iterate over an iterable.

        The iterator is executed in the host thread. The threads dynamically
        grab the elements. The iterator elements must hence be picklable to
        be transferred through the queue.

        If there is only one thread, no special operations are performed.
        Otherwise, effectively n-1 threads are used to process the iterable
        elements, and the host thread is used to provide them.

        You can specify a timeout for the clients to adhere.
        """
        self._assert_active()
        with self._queuelock:
            # Get this loop id.
            self._thread_loop_ids[self._thread_num] += 1
            loop_id = self._thread_loop_ids[self._thread_num]
            # Iterate.
            return _IterableQueueIterator(
                self._iter_queue, loop_id, self, iterable, element_timeout
            )


class _QueueIterator(object):

    """Iterator to create the dynamic schedule."""

    def __init__(self, queue, loop_id, pcontext):
        self._queue = queue
        self._loop_id = loop_id
        self._pcontext = pcontext

    # pylint: disable=non-iterator-returned
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        """Iterator implementation."""
        # pylint: disable=protected-access
        with self._pcontext._queuelock:
            # Check that the pool still deals with this loop.
            # pylint: disable=protected-access
            pool_loop_reached = max(self._pcontext._thread_loop_ids)
            if pool_loop_reached > self._loop_id or self._queue.empty():
                raise StopIteration()
            else:
                return self._queue.get()


class _IterableQueueIterator(object):

    """Iterator for the iterable queue."""

    def __init__(  # pylint: disable=too-many-arguments
        self, queue, loop_id, pcontext, iterable, element_timeout
    ):
        self._queue = queue
        self._loop_id = loop_id
        self._pcontext = pcontext
        if self._pcontext.thread_num == 0:
            self._iterable = iterable
            self._element_timeout = element_timeout

    # pylint: disable=non-iterator-returned
    def __iter__(self):
        if self._pcontext.num_threads == 1:
            return iter(self._iterable)
        return self

    def __next__(self):
        return self.next()

    def next(self):
        """Iterator implementation."""
        # pylint: disable=protected-access
        while True:
            if self._pcontext.thread_num == 0 and self._pcontext.num_threads > 1:
                # Producer.
                for iter_elem in self._iterable:
                    self._queue.put(iter_elem, timeout=self._element_timeout)
                for _ in range(self._pcontext.num_threads - 1):
                    self._queue.put(
                        "__queueend__:%d" % (self._pcontext._thread_loop_ids[0])
                    )
                raise StopIteration()
            elif self._pcontext.num_threads > 1:
                # Consumer.
                # Check that the pool still deals with this loop.
                # pylint: disable=protected-access
                with self._pcontext._queuelock:
                    pool_loop_reached = max(self._pcontext._thread_loop_ids)
                    master_reached = self._pcontext._thread_loop_ids[0]
                if pool_loop_reached > self._loop_id:
                    raise StopIteration()
                elif master_reached < self._loop_id:
                    # The producer did not reach this loop yet.
                    _time.sleep(.1)
                    continue
                else:
                    try:
                        queue_elem = self._queue.get(timeout=0.1)
                    except _Queue.Empty:
                        continue
                    if queue_elem == "__queueend__:%d" % self._loop_id:
                        raise StopIteration()
                    else:
                        return queue_elem
            else:
                # Single thread execution.
                # Should have never reached here, since this case is dealt with
                # in the __iter__ method!
                raise Exception("Internal error!")


def array(shape, dtype=_np.float64, autolock=False):                                           
    """Factory method for shared memory arrays supporting all numpy dtypes."""                 
    assert _NP_AVAILABLE, "To use the shared array object, numpy must be available!"           
    if not isinstance(dtype, _np.dtype):                                                       
        dtype = _np.dtype(dtype)                                                               
    # Not bothering to translate the numpy dtypes to ctype types directly,                     
    # because they're only partially supported. Instead, create a byte ctypes                  
    # array of the right size and use a view of the appropriate datatype.                      
    shared_arr = _multiprocessing.Array(                                                       
        "b", int(_np.prod(shape) * dtype.alignment), lock=autolock                             
    )                                                                                          
    with _warnings.catch_warnings():                                                           
        # For more information on why this is necessary, see                                   
        # https://www.reddit.com/r/Python/comments/j3qjb/parformatlabpool_replacement          
        _warnings.simplefilter("ignore", RuntimeWarning)                                       
        data = _np.ctypeslib.as_array(shared_arr).view(dtype).reshape(shape)                   
    return data


def zeros(shape, dtype=_np.float64, autolock=False):
    """
    Convenience method for creating a shared array with values initialized to zero.
    """
    arr = array(shape, dtype=dtype, autolock=autolock)
    arr[:] = 0.0
    return arr

def full(shape, val, dtype=_np.float64, autolock=False):
    """
    Convenience method for creating a shared array with values initialized to
    a specified constant value.
    """
    arr = array(shape, dtype=dtype, autolock=autolock)
    arr[:] = val
    return arr

