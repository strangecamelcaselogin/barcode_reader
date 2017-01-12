from barcode_reader import BarcodeReader


if __name__ == '__main__':
    # TODO threads
    # TODO Помехоустойчивое распознование штрихкода
    # TODO Алгоритм Брезенхейма
    # TODO для barcode normalize - отрезать лишнее
    reader = BarcodeReader()
    reader.run()
    #reader.one_frame('src/raw2.jpg')
