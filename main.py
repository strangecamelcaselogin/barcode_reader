from barcode_reader import BarcodeReader

ALPHABET = {
    #        0           1          2          3          4          5          6          7          8          9
    'L': ('0001101', '0011001', '0010011', '0111101', '0100011', '0110001', '0101111', '0111011', '0110111', '0001011'),
    'G': ('0100111', '0110011', '0011011', '0100001', '0011101', '0111001', '0000101', '0010001', '0001001', '0010111'),
    'R': ('1110010', '1100110', '1101100', '1000010', '1011100', '1001110', '1010000', '1000100', '1001000', '1110100')
}

FIRST = {
    0: 'LLLLLL',
    1: 'LLGLGG',
    2: 'LLGGLG',
    3: 'LLGGGL',
    4: 'LGLLGG',
    5: 'LGGLLG',
    6: 'LGGGLL',
    7: 'LGLGLG',
    8: 'LGLGGL',
    9: 'LGGLGL'
}


if __name__ == '__main__':
    # TODO threads
    # TODO Помехоустойчивое распознование штрихкода
    # TODO Алгоритм Брезенхейма
    # TODO для barcode normalize - отрезать лишнее
    reader = BarcodeReader()
    reader.run()
    #reader.one_frame('src/raw2.jpg')
