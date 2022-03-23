import numpy as np
import cv2
import yolo

# capture = yolo.load_capture()
#
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
#
# out = cv2.VideoWriter('test-write.avi', fourcc, 30.0, (1280, 720), True)


def wrapvideo(frame, ret):
    fourcc = cv2.VideoWriter.fourcc('F','L','V','I')
    out = cv2.VideoWriter('test-write.flv', fourcc, 30.0, (1280, 720), True)

    if ret:
        cv2.imshow('frame', frame)
        out.write(frame)
    out.release()
    # cv2.destroyAllWindows()


# capture.release()
# out.release()
# cv2.destroyAllWindows()

# if __name__ == "__main__":
#     pass
