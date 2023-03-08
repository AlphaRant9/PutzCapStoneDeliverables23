
DIGITS = '0123456789'

class Error:
    def __init__(self, error_name, details):
        self.error_name = error_name
        self.details = details
    
    def asString(self):
        result = f'{self.error_name}: {self.details}'
        return result

class IllegalCharError(Error):
    def __init__(self, details):
        super().__init__('Illegal Character', details)
    
    def asString(self):
        return super().asString()

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
        self.currentChar = None
        self.advance()

    def advance(self):
        self.pos += 1
        if (self.pos < len(self.text)): self.currentChar = self.text[self.pos]
        else: self.currentChar = None
            
    def makeTokens(self):
        tokens = []

        while self.currentChar != None:
            if (self.currentChar in ' \t'):
                self.advance()
            elif self.currentChar in DIGITS + '.':
                tokens.append(self.makeNumber())
            elif (self.currentChar == '+'):
                tokens.append(Token(TT_PLUS))
                self.advance()
            elif (self.currentChar == '-'):
                tokens.append(Token(TT_MINUS))
                self.advance()
            elif (self.currentChar == '*'):
                tokens.append(Token(TT_MUL))
                self.advance()
            elif (self.currentChar == '/'):
                tokens.append(Token(TT_DIV))
                self.advance()
            elif (self.currentChar == '('):
                tokens.append(Token(TT_LPAREN))
                self.advance()
            elif (self.currentChar == ')'):
                tokens.append(Token(TT_RPAREN))
                self.advance()
            else:
                char = self.currentChar
                self.advance()
                error = IllegalCharError("'" + char + "'")
                return [], error

        return tokens, None
    
    def makeNumber(self):
        numStr = ''
        dotCount = 0

        while self.currentChar != None and self.currentChar in DIGITS + '.':
            if self.currentChar == '.':
                if (dotCount == 1): break
                dotCount += 1
                numStr += '.'
            else:
                numStr += self.currentChar
            self.advance()

        if (dotCount == 0):
            return Token(TT_INT, int(numStr))
        else:
            return Token(TT_FLOAT, float(numStr))

def identifyTokens(text):
    lexer = Lexer(text)
    tokens, error = lexer.makeTokens()

    if error: print(error.asString())
    else: print(tokens)

    return tokens, error
