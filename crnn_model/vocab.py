from keras.layers import StringLookup
import tensorflow as tf

class Vocab():

    alphabets = "-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ"
    padding_token = 999

    char_to_num = StringLookup(vocabulary=list(alphabets), mask_token=None)

    num_to_char = StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    @staticmethod
    def vectorize_label(label, max_len):
        label = Vocab.char_to_num(tf.strings.unicode_split(
            label, input_encoding="UTF-8"))
        length = tf.shape(label)[0]
        pad_amount = max_len - length
        label = tf.pad(label, paddings=[[0, pad_amount]],
                    constant_values=Vocab.padding_token)
        return label
