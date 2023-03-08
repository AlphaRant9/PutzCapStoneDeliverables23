
DIGITS = '0123456789'

class Error:
    def __init__(self, error_name, details):
        self.error_name = error_name
        self.details = details
    
    def as_string(self):
        result = f'{self.error_name}: {self.details}'
        return result

class IllegalCharError(Error):
    def __init__(self, details):
        super().__init__('Illegal Character', details)
    
    def as_string(self):
        return super().as_string()

TT_INT = 'TT_INT'
TT_FLOAT = 'TT_FLOAT'
TT_PLUS = 'TT_PLUS'
TT_MINUS = 'TT_MINUS'
TT_MUL = 'TT_MUL'
TT_DIV = 'TT_DIV'
TT_LPAREN = 'TT_LPAREN'
TT_RPAREN = 'TT_RPAREN'

class Token:
    def __init__(self, type_, value=None):
        self.type = type_
        self.value = value
        
    def __repr__(self):
        if self.value != None: return f'{self.type}:{self.value}'
        else: return f'{self.type}'

class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = -1
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos += 1
        if (self.pos < len(self.text)): self.current_char = self.text[self.pos]
        else: self.current_char = None
            
    def make_tokens(self):
        tokens = []

        while self.current_char != None:
            if (self.current_char in ' \t'):
                self.advance()
            elif self.current_char in DIGITS + '.':
                tokens.append(self.make_number())
            elif (self.current_char == '+'):
                tokens.append(Token(TT_PLUS))
                self.advance()
            elif (self.current_char == '-'):
                tokens.append(Token(TT_MINUS))
                self.advance()
            elif (self.current_char == '*'):
                tokens.append(Token(TT_MUL))
                self.advance()
            elif (self.current_char == '/'):
                tokens.append(Token(TT_DIV))
                self.advance()
            elif (self.current_char == '('):
                tokens.append(Token(TT_LPAREN))
                self.advance()
            elif (self.current_char == ')'):
                tokens.append(Token(TT_RPAREN))
                self.advance()
            else:
                char = self.current_char
                self.advance()
                error = IllegalCharError("'" + char + "'")
                return [], error

        return tokens, None
    
    def make_number(self):
        num_str = ''
        dot_count = 0

        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if (dot_count == 1): break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()

        if (dot_count == 0):
            return Token(TT_INT, int(num_str))
        else:
            return Token(TT_FLOAT, float(num_str))

def run(text):
    lexer = Lexer(text)
    tokens, error = lexer.make_tokens()

    return tokens, error
