import cv2
import tensorflow as tf
import MTCNN.detect_face as detector

if __name__ == '__main__':
    sess = tf.Session()
    pnet, rnet, onet = detector.create_mtcnn(sess, None)
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    cap = cv2.VideoCapture(0)
    count = 0
    total = 0
    while (True):
        _, img = cap.read()
        im_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces, _ = detector.detect_face(im_, minsize, pnet, rnet, onet, threshold, factor)
        for box in faces:
            tmp = []
            for i in range(4):
                tmp.append(int(box[i]))
            cv2.rectangle(img, (tmp[0], tmp[1]), (tmp[2], tmp[3]), (255, 0, 0), 2)

        cv2.imshow('Face', img)
        if (cv2.waitKey(1) == 27):
            break

    cv2.destroyAllWindows()

