import cv2
import threading
import queue
from abc import ABC, abstractmethod
from typing import Optional


class AbstractFilter(ABC):
    def __init__(self, output_queue: queue.Queue) -> None:
        self.input_queue = queue.Queue()
        self.output_queue = output_queue
        self.worker_thread = threading.Thread(target=self.__worker, daemon=True)

    @abstractmethod
    def _process(self, frame: Optional[cv2.Mat]) -> Optional[cv2.Mat]:
        pass

    def __worker(self) -> None:
        frame_counter = 0
        while True:
            try:
                frame = self.input_queue.get()
                if frame is None:
                    self.output_queue.put(None)
                    break
                frame_counter += 1
                processed_frame = self._process(frame)

                print(f"{self.__class__.__name__} processed frame {frame_counter}")

                self.output_queue.put(processed_frame)
            except Exception as e:
                print(f"Error processing frame in {self.__class__.__name__}: {e}")

    def run(self) -> None:
        self.worker_thread = threading.Thread(target=self.__worker, daemon=True)
        self.worker_thread.start()

    def join(self) -> None:
        self.worker_thread.join()


# Filter 1: Black and White
class BlackAndWhiteFilter(AbstractFilter):
    _conversion_code = cv2.COLOR_BGR2GRAY

    def _process(self, frame: Optional[cv2.Mat]) -> Optional[cv2.Mat]:
        if frame is not None:
            return cv2.cvtColor(frame, self._conversion_code)
        return frame


# Filter 2: Mirror Effect
class MirrorFilter(AbstractFilter):
    _flip_code = 1

    def _process(self, frame):
        return cv2.flip(frame, self._flip_code)


# Filter 3: Resize Effect
class ResizeFilter(AbstractFilter):
    def __init__(self, output_queue, scale):
        super().__init__(output_queue)
        self.scale = scale

    def _process(self, frame):
        width = int(frame.shape[1] * self.scale)
        height = int(frame.shape[0] * self.scale)
        return cv2.resize(frame, (width, height))


# Filter 4: Color Inversion Effect
class InvertColorFilter(AbstractFilter):
    def _process(self, frame):
        return cv2.bitwise_not(frame)


# Filter 5 (additional one): Blur Effect
class BlurFilter(AbstractFilter):
    _blur_kernel = (3, 3)

    def _process(self, frame):
        return cv2.blur(frame, self._blur_kernel)


# Filter 6 (additional one): Gaussian Blur Effect
class GaussianBlurFilter(AbstractFilter):
    _gaussian_kernel = (15, 15)
    _sigma = 0

    def _process(self, frame):
        return cv2.GaussianBlur(frame, self._gaussian_kernel, self._sigma)


def capture_video_from_webcam(output_queue):
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Failed to find camera for video capture. Please verify that your camera is working.")

        while True:
            ret, frame = cap.read()
            if not ret:
                output_queue.put(None)
                break
            output_queue.put(frame)

        cap.release()

    except Exception as e:
        print("failed to capture video from web camera, error:", e)
        output_queue.put(None)


def read_video_from_file(output_queue):
    # put here path to your local video in the same format
    try:
        cap = cv2.VideoCapture(
            'C:\\Users\\sardi\\Videos\\Captures\\how to record screen windows 10 - Поиск в Google - Google Chrome '
            '2024-05-07 20-12-02.mp4')

        while True:
            ret, frame = cap.read()
            if not ret:
                output_queue.put(None)
                break
            output_queue.put(frame)

        cap.release()

    except Exception as e:
        print("failed to get video from file, error:", e)
        output_queue.put(None)


def display_video(input_queue, window_name):
    while True:
        frame = input_queue.get()
        if frame is None:
            break
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
    cv2.destroyWindow(window_name)


def main():
    """
    According to the lecture and BCK13, pipes-and-filters pattern requires separating filters
    into different asynchronous processes that interact with each other using thread-safe,
    interprocess communication channels (pipes).

    We have implemented this pattern by representing each filter as a separate object
    that accepts input and output channels during initialization. Filter object encapsulates
    the logic for processing frames, using an asynchronous worker thread that reads frames
    from the input channel, processes then using its algorithm, and sends them for further processing
    into the output channel. Filters has input (queue of frames that filter has to change)
    and output (queue in which filter puts changed frames) as communication channel between them.
    """

    # Queues for the pipe system
    # source queue - input frames from camera
    source_queue = queue.Queue()
    # sink queue - processed input frames from camera
    sink_queue = queue.Queue()

    resize_output_queue = queue.Queue()
    mirror_output_queue = queue.Queue()
    bnw_output_queue = queue.Queue()

    # Capturing and displaying the video runs on threads
    capture_thread = threading.Thread(target=capture_video_from_webcam, args=(source_queue,))
    # capture_thread = threading.Thread(target=read_video_from_file, args=(source_queue,))
    display_original_thread = threading.Thread(target=display_video, args=(source_queue, "Original Video"))
    display_processed_thread = threading.Thread(target=display_video, args=(sink_queue, "Processed Video"))

    # Filters (each running in its own thread)
    invert_filter = InvertColorFilter(output_queue=sink_queue)
    bnw_filter = BlackAndWhiteFilter(output_queue=invert_filter.input_queue)
    mirror_filter = MirrorFilter(output_queue=bnw_filter.input_queue)
    resize_filter = ResizeFilter(output_queue=mirror_filter.input_queue, scale=0.5)

    resize_filter.input_queue = source_queue

    # Start all threads (video and filters related ones)
    capture_thread.start()
    display_original_thread.start()
    display_processed_thread.start()
    resize_filter.run()
    mirror_filter.run()
    bnw_filter.run()
    invert_filter.run()

    # Join all threads to wait for completion
    # (Ensures that the main thread waits for the other threads to finish before continuing,
    # guaranteeing that all threads finish before the program terminates.)
    capture_thread.join()
    display_original_thread.join()
    display_processed_thread.join()
    resize_filter.join()
    mirror_filter.join()
    bnw_filter.join()
    invert_filter.join()


if __name__ == "__main__":
    main()
