import cv2
import threading
import queue


class Filter:
    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue):
        self.input_queue = input_queue
        self.output_queue = output_queue

    def process(self, frame):
        raise NotImplementedError("Subclasses should implement this!")

    def run(self):
        while True:
            frame = self.input_queue.get()
            if frame is None:
                self.output_queue.put(None)
                break
            processed_frame = self.process(frame)
            self.output_queue.put(processed_frame)


# Filter 1: Black and White
class BlackAndWhiteFilter(Filter):
    def process(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# Filter 2: Mirror Effect
class MirrorFilter(Filter):
    def process(self, frame):
        return cv2.flip(frame, 1)


# Filter 3: Resize Effect
class ResizeFilter(Filter):
    def __init__(self, input_queue, output_queue, scale):
        super().__init__(input_queue, output_queue)
        self.scale = scale

    def process(self, frame):
        width = int(frame.shape[1] * self.scale)
        height = int(frame.shape[0] * self.scale)
        return cv2.resize(frame, (width, height))


# Filter 4: Color Inversion Effect
class InvertColorFilter(Filter):
    def process(self, frame):
        return cv2.bitwise_not(frame)


# Filter 5 (additional one): Blur Effect
class BlurFilter(Filter):
    def process(self, frame):
        return cv2.blur(frame, (3, 3))


# Filter 6 (additional one): Gaussian Blur Effect
class GaussianBlurFilter(Filter):
    def process(self, frame):
        return cv2.GaussianBlur(frame, (15, 15), 0)


def capture_video_from_webcam(input_queue):
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Failed to find camera for video capture. Please verify that your camera is working.")

        while True:
            ret, frame = cap.read()
            if not ret:
                input_queue.put(None)
                break
            input_queue.put(frame)

        cap.release()

    except Exception as e:
        print(str(e))
        input_queue.put(None)


def capture_video_form_device(input_queue):
    # put here path to your video in the same format
    try:
        cap = cv2.VideoCapture(
            'C:\\Users\\sardi\\Videos\\Captures\\how to record screen windows 10 - Поиск в Google - Google Chrome '
            '2024-05-07 20-12-02.mp4')

        while True:
            ret, frame = cap.read()
            if not ret:
                input_queue.put(None)
                break
            input_queue.put(frame)

        cap.release()

    except Exception as e:
        print(str(e))
        input_queue.put(None)


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
    So, as lecturer said in the video and what was stated in the book. We have implemented this pattern such that
    each filters works on different processes and processes (filters) exchange data through queues (pipes).

    Each filter has input (queue of frames that filter has to change) and output (queue in which filter puts changed
    frames)
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
    display_original_thread = threading.Thread(target=display_video, args=(source_queue, "Original Video"))
    display_processed_thread = threading.Thread(target=display_video, args=(sink_queue, "Processed Video"))

    # Filters (each running in its own thread)
    resize_filter = ResizeFilter(input_queue=source_queue, output_queue=resize_output_queue, scale=0.5)
    mirror_filter = MirrorFilter(input_queue=resize_output_queue, output_queue=mirror_output_queue)
    bnw_filter = BlackAndWhiteFilter(input_queue=mirror_output_queue, output_queue=bnw_output_queue)
    invert_filter = InvertColorFilter(input_queue=bnw_output_queue, output_queue=sink_queue)

    resize_thread = threading.Thread(target=resize_filter.run)
    mirror_thread = threading.Thread(target=mirror_filter.run)
    bnw_thread = threading.Thread(target=bnw_filter.run)
    invert_thread = threading.Thread(target=invert_filter.run)

    # Start all threads (video and filters related ones)
    capture_thread.start()
    display_original_thread.start()
    display_processed_thread.start()
    resize_thread.start()
    mirror_thread.start()
    bnw_thread.start()
    invert_thread.start()


if __name__ == "__main__":
    main()
