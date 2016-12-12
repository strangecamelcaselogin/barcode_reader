from collections import Counter  # , OrderedDict, namedtuple
from copy import deepcopy as deepcopy
from time import time

#  import numpy as np
import cv2
import scipy.signal as scs
import matplotlib.pyplot as plt

from settings import *


class BarcodeReader:
    def __init__(self):
        self.gc = self.gauss_core(k=15, sigma=1)
        self.lines = []
        self.width, self.height = 0, 0

    @staticmethod
    def gauss_core(k, sigma):
        gc = scs.gaussian(k, std=sigma, sym=True)
        gc /= sum(gc)

        return gc

    @staticmethod
    def plot_data(data):
        plt.plot(data['line'], 'b')
        plt.plot(data['derivative'], 'r')
        plt.plot(data['debug'], [0 for i in range(len(data['debug']))], '.')

        plt.plot([data['threshold'] for i in range(len(data['line']))], 'g--')
        plt.plot([-data['threshold'] for i in range(len(data['line']))], 'g--')

        plt.grid(True)
        plt.show()

    @staticmethod
    def barcode_human_repr(barcode):
        swap = (0, 0, 1)
        bit = sum(round(ln) for _, ln in barcode)

        str_ = str(bit) + ' | '
        for type_, ln in barcode:
            # print(str(swap[type_]) * ln, end=' ')
            str_ += str(str(swap[type_]) * round(ln)) + ' '

        return str_

    @staticmethod
    def get_line(frame, size, p1=(0, 0), p2=(0, 0)):
        # Получение линии с кадра
        frame_width, frame_high = size
        x1, y1 = p1
        x2, y2 = p2

        if y1 == y2:
            return deepcopy(frame[y1][x1:x2])

        else:
            return None

    @staticmethod
    def get_derivative(line):
        length = len(line)
        line = [int(l) for l in line]
        der_line = [0 for _ in range(length)]
        for i in range(1, length):
            der_line[i] = line[i - 1] - line[i]

        return der_line

    @staticmethod
    def get_raw_barcode(derivative, threshold):
        """
        Ищем отрезок, образованный порогом на производной,
        считаем его середину, ищем следующий и его середину
        и вычисляем расстояние между ними, их тип (черный белый)

        черный == -1
        белый == 1
        """

        raw_barcode = []
        peak, prev_peak = 0, 0  # Точки пика и предыдущего пика
        start, end = 0, 0  # Точка начала и конца роста производной выше порога
        peak_state = 0
        find_peak = False

        peaks_for_debug = []
        p_value = 0
        for index, value in enumerate(derivative):
            if abs(value) > threshold:  # Если мы наткнулись на рост производной выше порога.
                if find_peak:  # В очередной раз.
                    end += 1

                else:  # В первый раз.
                    find_peak = True
                    start = index
                    end = index
                    peak_state = 1 if value > 0 else -1

            else:  # Если значение  производной не превышает порога.
                find_peak = False
                if abs(p_value) > threshold:
                    peak = (index - 1) - round(((end + 1) - start) / 2)  # MAGIC round - int, +1 +0
                    peaks_for_debug.append(peak)
                    if prev_peak != 0:
                        raw_barcode.append((peak_state, peak - prev_peak + 1))

                    prev_peak = peak

            p_value = value

        return raw_barcode, peaks_for_debug

    @staticmethod
    def barcode_normalize(raw_barcode):
        """
        Нормализация - берем самую частую длину линии одного из цветов и делим
        все линии такого цвета на нее.
        """

        blist = []
        wlist = []
        for t, ln in raw_barcode:
            wlist.append(ln) if t == 1 else blist.append(ln)

        md_black = Counter(blist).most_common(1)[0][0]
        md_white = Counter(wlist).most_common(1)[0][0]
        med = (md_black + md_white) / 2

        most_common = (0, med, md_black)

        reset_len = max(*most_common) * 5
        norm_barcode = []
        for type_, ln in raw_barcode:  # Нормализация
            if ln < reset_len and (ln * 2) > most_common[type_]:
                norm_barcode.append((type_, round(ln / most_common[type_])))

        return norm_barcode, most_common

    @staticmethod
    def barcode_read(barcode):
        # Просмотреть длинну.
        # Группы 101 01010 и 101.
        return True

    def frame_analysis(self, frame, debug=False):
        """
        Перебираем переданный набор линий, строим для каждой производную,
        и опуская порог ищем хотя бы 59 всплесков на ней.

        Нормалиуем.

        Пытаемся считать.

        frame -> line -> derivative -> raw_barcode -> barcode (нормированный) -> result
        """
        size = self.width, self.height
        wave_cnt = 0
        timer = time()
        for p1, p2 in self.lines:
            line = self.get_line(frame, size, p1, p2)
            cnv_derivative = scs.convolve(self.get_derivative(line), self.gc, mode='same')

            threshold = DEF_THRESHOLD
            for index in range(DEF_TRY_CNT):
                raw_barcode, dbg = self.get_raw_barcode(cnv_derivative, threshold)
                raw_wave_cnt = len(raw_barcode)

                if raw_wave_cnt >= DEF_BARCODE_LEN:

                    barcode, m_common = self.barcode_normalize(raw_barcode)
                    wave_cnt = len(barcode)

                    if wave_cnt >= DEF_BARCODE_LEN:  # Если на данной итерации хотя бы 59 всплесков, то продолжим
                        # print('{}: порог - {}, выбросов на raw - {}, выбросов norm - {}'.format(index, threshold,
                        #                                                                        raw_wave_cnt, wave_cnt))

                        if debug:
                            print('most_common:', m_common)
                            print('raw:', raw_barcode)
                            print('norm:', barcode)

                        result = self.barcode_read(barcode)
                        if result:
                            return {'line': line,
                                    'derivative': cnv_derivative,
                                    'barcode': barcode,
                                    'result': result,
                                    'debug': dbg,
                                    'threshold': threshold,
                                    'time': time() - timer}

                threshold -= THRESHOLD_STEP
                # print('{}: порог - {}, выбросов на raw - {}, выбросов norm - {}'.format(index, threshold, raw_wave_cnt, wave_cnt))

        return None

    def run(self):
        # Вызываем frame_analysis для каждого кадра,
        # если вернула не None, значит нашла штрихкод

        cap = cv2.VideoCapture(0)
        ret, _ = cap.read()
        if not ret:
            raise Exception('Capture device error.')

        self.width, self.height = int(cap.get(3)), int(cap.get(4))
        print('camera resolution: {}x{}'.format(self.width, self.height))

        self.lines = [((0, round(self.height / 2)), (self.width, round(self.height / 2)))]

        start_time = time()
        fps = 0
        frame_cnt = 0
        while cap.isOpened():
            _, frame = cap.read()  # Получаем ЦВЕТНОЙ кадр с камеры
            colored_frame = cv2.flip(frame.copy(), 1)  # Отзеркалить
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Переводим в серый

            data = self.frame_analysis(frame)

            if data is not None:
                print(self.barcode_human_repr(data['barcode']))
                self.plot_data(data)

                cv2.waitKey(0)

                start_time = time()

            for p1, p2 in self.lines:
                cv2.line(colored_frame, p1, p2, color=[0, 0, 255])

            cv2.putText(colored_frame, "FPS: {}".format(fps), (10, 20), cv2.FONT_HERSHEY_PLAIN,
                        fontScale=0.75, color=[0, 255, 255], lineType=cv2.LINE_AA)

            cv2.imshow('frame', colored_frame)

            frame_cnt += 1
            fps = round(frame_cnt / (time() - start_time))

            if cv2.waitKey(1) == 27:  # cv2,waitKey(1) & 0xFF == ord('q'):
                break



        cap.release()
        cv2.destroyAllWindows()

    def one_frame(self, filename):

        img = cv2.imread(filename, 0)  # 0 - grayscale
        cv2.namedWindow("frame", cv2.WINDOW_KEEPRATIO)

        self.width, self.height = len(img[0]), len(img)
        print('frame size: {}x{}\n'.format(self.width, self.height))

        self.lines = [((0, round(self.height / 2)), (self.width, round(self.height / 2)))]

        data = self.frame_analysis(img, debug=True)

        if data:
            print('SUCCESS')
            print('frame analysis by {} sec'.format(data['time']))

            print(self.barcode_human_repr(data['barcode']))

            for p1, p2 in self.lines:
                cv2.line(img, p1, p2, color=255, thickness=3)

            cv2.imshow('frame', img)
            self.plot_data(data)
            cv2.destroyAllWindows()

        else:
            print('Штрихкод не найден')
