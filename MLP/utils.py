from layer import Layer, FC_Layer, Tanh, ReLU, Sigmoid, Softmax
def create_activation_function(activation_func):
    if activation_func == 'tanh':
        return Tanh()
    elif activation_func == 'relu':
        return ReLU()
    elif activation_func == 'sigmoid':
        return Sigmoid()
    elif activation_func == 'softmax':
        return Softmax()
    else:
        return None
    
def create_char_level_tokenizer(text: str ):
    """Tạo một tokenizer cho văn bản ở cấp ký tự"""
    class CharTokenizer():
        def __init__(self, text: str):
            self.chars = sorted(list(set(text)))# set để lấy các ký tự duy nhất , có thể bỏ list(), lúc này sorterd tự động ép kiểu về list để trả ra list 
            self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
            self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
            self.vocab_size = len(self.chars)

        def fit(self, text: str):
            
            return [self.char2idx[ch] for ch in text]

        def inverse(self, idx: list):
            return ''.join([self.idx2char[i] for i in idx])

        def __len__(self):
            return self.vocab_size
    return CharTokenizer(text)
        
        
# if __name__ == "__main__":
#     text = "Hello, world!"
#     tokenizer = create_char_level_tokenizer(text)
#     print(tokenizer.chars)
#     print(tokenizer.char2idx)
#     print(tokenizer.idx2char)
#     print(tokenizer.fit(text))
#     print(tokenizer.inverse([0, 1, 2, 3, 4]))