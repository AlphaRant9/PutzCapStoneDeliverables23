import string

from stringsWithArrows import *

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS


class Error:
    def __init__(self, posStart, posEnd, errorName, details):
        self.posStart = posStart
        self.posEnd = posEnd
        self.errorName = errorName
        self.details = details

    def asString(self):
        result = f'{self.errorName}: {self.details}\n'
        result += f'File {self.posStart.fn}, line {self.posStart.ln + 1}'
        result += '\n\n' + stringsWithArrows(self.posStart.ftxt, self.posStart, self.posEnd)
        return result


class IllegalCharError(Error):
    def __init__(self, posStart, posEnd, details):
        super().__init__(posStart, posEnd, 'Illegal Character', details)


class InvalidSyntaxError(Error):
    def __init__(self, posStart, posEnd, details=''):
        super().__init__(posStart, posEnd, 'Invalid Syntax', details)


class RTError(Error):
    def __init__(self, posStart, posEnd, details, context):
        super().__init__(posStart, posEnd, 'Runtime Error', details)
        self.context = context

    def asString(self):
        result = self.generateTraceback()
        result += f'{self.errorName}: {self.details}'
        result += '\n\n' + stringsWithArrows(self.posStart.ftxt, self.posStart, self.posEnd)
        return result

    def generateTraceback(self):
        result = ''
        pos = self.posStart
        ctx = self.context

        while ctx:
            result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.displayName}\n' + result
            pos = ctx.parentEntryPos
            ctx = ctx.parent

        return 'Traceback (most recent call last):\n' + result


class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, currentChar=None):
        self.idx += 1
        self.col += 1

        if currentChar == '\n':
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)


TT_INT = 'INT'
TT_FLOAT = 'FLOAT'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_POW = 'POW'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_EOF = 'EOF'
TT_KEYWORD = 'KEYWORD'
TT_IDENTIFIER = 'IDENTIFIER'
TT_EQ = 'EQ'

KEYWORDS = [

    'var'

]


class Token:
    def __init__(self, type_, value=None, posStart=None, posEnd=None):
        self.type = type_
        self.value = value

        if posStart:
            self.posStart = posStart.copy()
            self.posEnd = posStart.copy()
            self.posEnd.advance()

        if posEnd:
            self.posEnd = posEnd

    def matches(self, type_, value):
        return self.type == type_ and self.value == value

    def __repr__(self):
        if self.value:
            return f'{self.type}:{self.value}'
        return f'{self.type}'


class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.currentChar = None
        self.advance()

    def advance(self):
        self.pos.advance(self.currentChar)
        self.currentChar = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def makeTokens(self):
        tokens = []

        while self.currentChar is not None:
            if self.currentChar in ' \t':
                self.advance()
            elif self.currentChar in DIGITS:
                tokens.append(self.makeNumber())
            elif self.currentChar.lower() in LETTERS:
                tokens.append(self.makeIdentifier())
            elif self.currentChar == '+':
                tokens.append(Token(TT_PLUS, posStart=self.pos))
                self.advance()
            elif self.currentChar == '-':
                tokens.append(Token(TT_MINUS, posStart=self.pos))
                self.advance()
            elif self.currentChar == '*':
                tokens.append(Token(TT_MUL, posStart=self.pos))
                self.advance()
            elif self.currentChar == '/':
                tokens.append(Token(TT_DIV, posStart=self.pos))
                self.advance()
            elif self.currentChar == '^':
                tokens.append(Token(TT_POW, posStart=self.pos))
                self.advance()
            elif self.currentChar == '(':
                tokens.append(Token(TT_LPAREN, posStart=self.pos))
                self.advance()
            elif self.currentChar == ')':
                tokens.append(Token(TT_RPAREN, posStart=self.pos))
                self.advance()
            elif self.currentChar == '=':
                tokens.append(Token(TT_EQ, posStart=self.pos))
                self.advance()
            else:
                posStart = self.pos.copy()
                char = self.currentChar
                self.advance()
                return [], IllegalCharError(posStart, self.pos, "'" + char + "'")

        tokens.append(Token(TT_EOF, posStart=self.pos))
        return tokens, None

    def makeNumber(self):
        numStr = ''
        dotCount = 0
        posStart = self.pos.copy()

        while self.currentChar is not None and self.currentChar in DIGITS + '.':
            if self.currentChar == '.':
                if dotCount == 1: break
                dotCount += 1
                numStr += '.'
            else:
                numStr += self.currentChar
            self.advance()

        if dotCount == 0:
            return Token(TT_INT, int(numStr), posStart, self.pos)
        else:
            return Token(TT_FLOAT, float(numStr), posStart, self.pos)

    def makeIdentifier(self):
        idStr = ''
        posStart = self.pos.copy()

        while self.currentChar is not None and self.currentChar in LETTERS_DIGITS + '_':
            idStr += self.currentChar
            self.advance()

        tokType = TT_KEYWORD if idStr in KEYWORDS else TT_IDENTIFIER
        return Token(tokType, idStr, posStart, self.pos)


class NumberNode:
    def __init__(self, tok):
        self.tok = tok

        self.posStart = self.tok.posStart
        self.posEnd = self.tok.posEnd

    def __repr__(self):
        return f'{self.tok}'


class VarAccessNode:
    def __init__(self, varNameTok):
        self.varNameTok = varNameTok

        self.posStart = self.varNameTok.posStart
        self.posEnd = self.varNameTok.posEnd


class VarAssignNode:
    def __init__(self, varNameTok, valueNode):
        self.varNameTok = varNameTok
        self.valueNode = valueNode

        self.posStart = self.varNameTok.posStart
        self.posEnd = self.valueNode.posEnd


class BinOpNode:
    def __init__(self, leftNode, opTok, rightNode):
        self.leftNode = leftNode
        self.opTok = opTok
        self.rightNode = rightNode

        self.posStart = self.leftNode.posStart
        self.posEnd = self.rightNode.posEnd

    def __repr__(self):
        return f'({self.leftNode}, {self.opTok}, {self.rightNode})'


class UnaryOpNode:
    def __init__(self, opTok, node):
        self.opTok = opTok
        self.node = node

        self.posStart = self.opTok.posStart
        self.posEnd = node.posEnd

    def __repr__(self):
        return f'({self.opTok}, {self.node})'


class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None

    def registerAdvancement(self):
        pass

    def register(self, res):
        if res.error:
            self.error = res.error
        return res.node

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        self.error = error
        return self


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.tokIdx = -1
        self.advance()

    def advance(self, ):
        self.tokIdx += 1
        if self.tokIdx < len(self.tokens):
            self.currentTok = self.tokens[self.tokIdx]
        return self.currentTok

    def parse(self):
        res = self.expr()
        if not res.error and self.currentTok.type != TT_EOF:
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                "Expected '+', '-', '*' or '/'"
            ))
        return res

    def atom(self):
        res = ParseResult()
        tok = self.currentTok

        if tok.type in (TT_INT, TT_FLOAT):
            res.registerAdvancement()
            self.advance()
            return res.success(NumberNode(tok))

        elif tok.type == TT_IDENTIFIER:
            res.registerAdvancement()
            self.advance()
            return res.success(VarAccessNode(tok))

        elif tok.type == TT_LPAREN:
            res.registerAdvancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.currentTok.type == TT_RPAREN:
                res.registerAdvancement()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
                    self.currentTok.posStart, self.currentTok.posEnd,
                    "Expected ')'"
                ))

        return res.failure(InvalidSyntaxError(
            tok.posStart, tok.posEnd,
            "Expected int, float, identifier, '+', '-' or '('"
        ))

    def power(self):
        return self.binOp(self.atom, (TT_POW,), self.factor)

    def factor(self):
        res = ParseResult()
        tok = self.currentTok

        if tok.type in (TT_PLUS, TT_MINUS):
            res.registerAdvancement()
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok, factor))

        return self.power()

    def term(self):
        return self.binOp(self.factor, (TT_MUL, TT_DIV))

    def expr(self):

        res = ParseResult()

        if self.currentTok.matches(TT_KEYWORD, 'var'):
            res.registerAdvancement()
            self.advance()

            if self.currentTok.type is not TT_IDENTIFIER:
                return res.failure(InvalidSyntaxError(self.currentTok.posStart,
                                                      self.currentTok.posEnd,
                                                      "Expected Identifier"))

            varName = self.currentTok
            res.registerAdvancement()
            self.advance()

            if self.currentTok.type is not TT_EQ:
                return res.failure(
                    InvalidSyntaxError(self.currentTok.posStart,
                                       self.currentTok.posEnd,
                                       "Expected '='"))

            res.registerAdvancement()
            self.advance()
            expr = res.register(self.expr())

            if res.error:
                return res

            return res.success(VarAssignNode(varName, expr))

        node = res.register(self.binOp(self.term, (TT_PLUS, TT_MINUS)))

        if res.error:
            return res.failure(
                InvalidSyntaxError(
                    self.currentTok.posStart,
                    self.currentTok.posEnd,
                    "Expected 'var', int, float, identifier, '+', '-', or '('"
                )
            )

        return res.success(node)

    def binOp(self, funcA, ops, funcB=None):
        if funcB == None:
            funcB = funcA

        res = ParseResult()
        left = res.register(funcA())
        if res.error: return res

        while self.currentTok.type in ops:
            opTok = self.currentTok
            res.registerAdvancement()
            self.advance()
            right = res.register(funcB())
            if res.error: return res
            left = BinOpNode(left, opTok, right)

        return res.success(left)


class RTResult:
    def __init__(self):
        self.value = None
        self.error = None

    def register(self, res):
        if res.error: self.error = res.error
        return res.value

    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self


class Number:
    def __init__(self, value):
        self.value = value
        self.setPos()
        self.setContext()

    def setPos(self, posStart=None, posEnd=None):
        self.posStart = posStart
        self.posEnd = posEnd
        return self

    def setContext(self, context=None):
        self.context = context
        return self

    def addedTo(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).setContext(self.context), None

    def subbedBy(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).setContext(self.context), None

    def multedBy(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).setContext(self.context), None

    def divedBy(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.posStart, other.posEnd,
                    'Division by zero',
                    self.context
                )

            return Number(self.value / other.value).setContext(self.context), None

    def powedBy(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).setContext(self.context), None

    def __repr__(self):
        return str(self.value)


class Context:
    def __init__(self, displayName, parent=None, parentEntryPos=None):
        self.displayName = displayName
        self.parent = parent
        self.parentEntryPos = parentEntryPos
        self.symbolTable = None


class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None

    def get(self, name):
        value = self.symbols.get(name, None)
        if value is None and self.parent:
            return self.parent.get(name)

        return value

    def set(self, name, value):
        self.symbols[name] = value

    def remove(self, name):
        del self.symbols[name]


class Interpreter:
    def visit(self, node, context):
        methodName = f'visit{type(node).__name__}'
        method = getattr(self, methodName, self.noVisitMethod)
        return method(node, context)

    def noVisitMethod(self, node, context):
        raise Exception(f'No visit{type(node).__name__} method defined')

    def visitNumberNode(self, node, context):
        return RTResult().success(
            Number(node.tok.value).setContext(context).setPos(node.posStart, node.posEnd)
        )

    def visitVarAccessNode(self, node, context):
        res = RTResult()
        varName = node.varNameTok.value
        value = context.symbolTable.get(varName)

        if not value:
            return res.failure(
                RTError(
                    node.posStart,
                    node.posEnd,
                    f"'{varName}' is not defined",
                    context
                )
            )

        return res.success(value)

    def visitVarAssignNode(self, node, context):
        res = RTResult()
        varName = node.varNameTok.value
        visit = (self.visit(node.valueNode, context))
        value = res.register(visit)

        if res.error:
            return res

        context.symbolTable.set(varName, value)
        return res.success(value)

    def visitBinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.leftNode, context))
        if res.error:
            return res
        right = res.register(self.visit(node.rightNode, context))
        if res.error:
            return res

        if node.opTok.type == TT_PLUS:
            result, error = left.addedTo(right)
        elif node.opTok.type == TT_MINUS:
            result, error = left.subbedBy(right)
        elif node.opTok.type == TT_MUL:
            result, error = left.multedBy(right)
        elif node.opTok.type == TT_DIV:
            result, error = left.divedBy(right)
        elif node.opTok.type == TT_POW:
            result, error = left.powedBy(right)

        if error:
            return res.failure(error)
        else:
            return res.success(result.setPos(node.posStart, node.posEnd))

    def visitUnaryOpNode(self, node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.error: return res

        error = None

        if node.opTok.type == TT_MINUS:
            number, error = number.multedBy(Number(-1))

        if error:
            return res.failure(error)
        else:
            return res.success(number.setPos(node.posStart, node.posEnd))


globalSymbolTable = SymbolTable()
globalSymbolTable.set("null", Number(0))


def run(fn, text):
    lexer = Lexer(fn, text)
    tokens, error = lexer.makeTokens()
    if error:
        return None, error

    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error:
        return None, ast.error

    interpreter = Interpreter()
    context = Context('<program>')
    context.symbolTable = globalSymbolTable
    result = interpreter.visit(ast.node, context)

    return result.value, result.error
