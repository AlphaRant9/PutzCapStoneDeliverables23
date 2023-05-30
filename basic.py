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


class ExpectedCharError(Error):
    def __init__(self, posStart, posEnd, details):
        super().__init__(posStart, posEnd, 'Expected Character', details)


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
TT_EE = 'EE'
TT_NE = 'NE'
TT_LT = 'LT'
TT_GT = 'GT'
TT_LTE = 'LTE'
TT_GTE = 'GTE'

KEYWORDS = [

    'var',
    'and',
    'or',
    'not',
    'if',
    'then',
    'elif',
    'else',
    'for',
    'to',
    'step',
    'while'

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
            elif self.currentChar == '!':
                tok, error = self.makeNotEquals()
                if error: return [], error
                tokens.append(tok)
            elif self.currentChar == '=':
                tokens.append(self.makeEquals())
            elif self.currentChar == '<':
                tokens.append(self.makeLessThan())
            elif self.currentChar == '>':
                tokens.append(self.makeGreaterThan())
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

    def makeNotEquals(self):
        posStart = self.pos.copy()
        self.advance()

        if self.currentChar == '=':
            self.advance()
            return Token(TT_NE, posStart=posStart, posEnd=self.pos), None

        self.advance()
        return None, ExpectedCharError(posStart, self.pos, "'=' (after '!')")

    def makeEquals(self):
        tokType = TT_EQ
        posStart = self.pos.copy()
        self.advance()

        if self.currentChar == '=':
            self.advance()
            tokType = TT_EE

        return Token(tokType, posStart=posStart, posEnd=self.pos)

    def makeLessThan(self):
        tokType = TT_LT
        posStart = self.pos.copy()
        self.advance()

        if self.currentChar == '=':
            self.advance()
            tokType = TT_LTE

        return Token(tokType, posStart=posStart, posEnd=self.pos)

    def makeGreaterThan(self):
        tokType = TT_GT
        posStart = self.pos.copy()
        self.advance()

        if self.currentChar == '=':
            self.advance()
            tokType = TT_GTE

        return Token(tokType, posStart=posStart, posEnd=self.pos)


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


class IfNode:

    def __init__(self, cases, elseCase):
        self.cases = cases
        self.elseCase = elseCase

        self.posStart = cases[0][0].posStart
        self.posEnd = (self.elseCase or self.cases[len(self.cases) - 1][0]).posEnd


class ForNode:
    def __init__(self, varNameTok, startValueNode, endValueNode, stepValueNode, bodyNode):
        self.varNameTok = varNameTok
        self.startValueNode = startValueNode
        self.endValueNode = endValueNode
        self.stepValueNode = stepValueNode
        self.bodyNode = bodyNode

        self.posStart = self.varNameTok.posStart
        self.posEnd = self.bodyNode.posEnd


class WhileNode:
    def __init__(self, conditionNode, bodyNode):
        self.conditionNode = conditionNode
        self.bodyNode = bodyNode

        self.posStart = self.conditionNode.posStart
        self.posEnd = self.bodyNode.posEnd


class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advanceCount = 0

    def registerAdvancement(self):
        self.advanceCount += 1

    def register(self, res):
        self.advanceCount += res.advanceCount
        if res.error:
            self.error = res.error
        return res.node

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        if not self.error or self.advanceCount == 0:
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

        elif tok.matches(TT_KEYWORD, 'if'):
            ifExpr = res.register(self.ifExpr())
            if res.error:
                return res
            return res.success(ifExpr)
        elif tok.matches(TT_KEYWORD, 'for'):
            forExpr = res.register(self.forExpr())
            if res.error:
                return res
            return res.success(forExpr)
        elif tok.matches(TT_KEYWORD, 'while'):
            whileExpr = res.register(self.whileExpr())
            if res.error:
                return res
            return res.success(whileExpr)

        return res.failure(InvalidSyntaxError(
            tok.posStart, tok.posEnd,
            "Expected int, float, identifier, '+', '-' or '('"
        ))

    def ifExpr(self):
        res = ParseResult()
        cases = []
        elseCase = None

        if not self.currentTok.matches(TT_KEYWORD, 'if'):
            return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected 'if'"))

        res.registerAdvancement()
        self.advance()
        condition = res.register(self.expr())
        if res.error:
            return res

        if not self.currentTok.matches(TT_KEYWORD, 'then'):
            return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected 'then'"))

        res.registerAdvancement()
        self.advance()

        expr = res.register(self.expr())
        if res.error:
            return res
        cases.append((condition, expr))

        while self.currentTok.matches(TT_KEYWORD, 'elif'):
            res.registerAdvancement()
            self.advance()
            condition = res.register(self.expr())
            if res.error:
                return res

            if not self.currentTok.matches(TT_KEYWORD, 'then'):
                return res.failure(
                    InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd, f"Expected 'then'"))

            res.registerAdvancement()
            self.advance()

            expr = res.register(self.expr())
            if res.error:
                return res
            cases.append((condition, expr))

        if self.currentTok.matches(TT_KEYWORD, 'else'):
            res.registerAdvancement()
            self.advance()

            elseCase = res.register(self.expr())
            if res.error:
                return res

            # elseCase = expr

        return res.success(IfNode(cases, elseCase))

    def forExpr(self):
        res = ParseResult()

        if not self.currentTok.matches(TT_KEYWORD, 'for'):
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"Expected 'for'"
            ))

        res.registerAdvancement()
        self.advance()

        if self.currentTok.type != TT_IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"Expected identifier"
            ))

        varName = self.currentTok
        res.registerAdvancement()
        self.advance()

        if self.currentTok.type != TT_EQ:
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"Expected '='"
            ))

        res.registerAdvancement()
        self.advance()

        startValue = res.register(self.expr())
        if res.error:
            return res

        if not self.currentTok.matches(TT_KEYWORD, 'to'):
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"Expected 'to'"
            ))

        res.registerAdvancement()
        self.advance()

        endValue = res.register(self.expr())
        if res.error:
            return res

        if self.currentTok.matches(TT_KEYWORD, 'step'):
            res.registerAdvancement()
            self.advance()

            stepValue = res.register(self.expr())
            if res.error:
                return res

        else:
            stepValue = None

        if not self.currentTok.matches(TT_KEYWORD, 'then'):
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"Expected 'then'"
            ))

        res.registerAdvancement()
        self.advance()

        body = res.register(self.expr())
        if res.error:
            return res

        return res.success(ForNode(varName, startValue, endValue, stepValue, body))

    def whileNode(self):
        res = ParseResult()

        if not self.currentTok.matches(TT_KEYWORD, 'while'):
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"Expected 'while'"
            ))

        res.registerAdvancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error:
            return res

        if not self.currentTok.matches(TT_KEYWORD, 'then'):
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"Expected 'then'"
            ))

        res.registerAdvancement()
        self.advance()

        body = res.register(self.expr())
        if res.error:
            return res

        return res.success(WhileNode(condition, body))

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

    def arithExpr(self):
        return self.binOp(self.term, (TT_PLUS, TT_MINUS))

    def compExpr(self):
        res = ParseResult()

        if self.currentTok.matches(TT_KEYWORD, 'not'):
            opTok = self.currentTok
            res.registerAdvancement()
            self.advance()

            node = res.register(self.compExpr())
            if res.error: return res
            return res.success(UnaryOpNode(opTok, node))

        node = res.register(self.binOp(self.arithExpr, (TT_EE, TT_NE, TT_LT, TT_LTE, TT_GT, TT_GTE)))

        if res.error:
            return res.failure(InvalidSyntaxError(self.currentTok.posStart, self.currentTok.posEnd,
                                                  "Expected int, float, identifier, '+', '-', '(', or 'not'"))

        return res.success(node)

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

        node = res.register(self.binOp(self.compExpr, ((TT_KEYWORD, "and"), (TT_KEYWORD, "or"))))

        if res.error:
            return res.failure(
                InvalidSyntaxError(
                    self.currentTok.posStart,
                    self.currentTok.posEnd,
                    "Expected 'var', int, float, identifier, '+', '-', '(', or 'not'"
                )
            )

        return res.success(node)

    def binOp(self, funcA, ops, funcB=None):
        if funcB is None:
            funcB = funcA

        res = ParseResult()
        left = res.register(funcA())
        if res.error:
            return res

        while self.currentTok.type in ops or (self.currentTok.type, self.currentTok.value) in ops:
            opTok = self.currentTok
            res.registerAdvancement()
            self.advance()
            right = res.register(funcB())
            if res.error:
                return res
            left = BinOpNode(left, opTok, right)

        return res.success(left)


class RTResult:
    def __init__(self):
        self.value = None
        self.error = None

    def register(self, res):
        if res.error:
            self.error = res.error
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

    def getComparisonEq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).setContext(self.context), None

    def getComparisonNe(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).setContext(self.context), None

    def getComparisonLt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).setContext(self.context), None

    def getComparisonGt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).setContext(self.context), None

    def getComparisonLte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).setContext(self.context), None

    def getComparisonGte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).setContext(self.context), None

    def andedBy(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).setContext(self.context), None

    def oredBy(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).setContext(self.context), None

    def notted(self):
        return Number(1 if self.value == 0 else 0).setContext(self.context), None

    def copy(self):
        copy = Number(self.value)
        copy.setPos(self.posStart, self.posEnd)
        copy.setContext(self.context)
        return copy

    def isTrue(self):
        return self.value != 0

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

        value = value.copy().setPos(node.posStart, node.posEnd)
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
        elif node.opTok.type == TT_EE:
            result, error = left.getComparisonEq(right)
        elif node.opTok.type == TT_NE:
            result, error = left.getComparisonNq(right)
        elif node.opTok.type == TT_LT:
            result, error = left.getComparisonLt(right)
        elif node.opTok.type == TT_GT:
            result, error = left.getComparisonGt(right)
        elif node.opTok.type == TT_LTE:
            result, error = left.getComparisonLte(right)
        elif node.opTok.type == TT_GTE:
            result, error = left.getComparisonGte(right)
        elif node.opTok.matches(TT_KEYWORD, 'and'):
            result, error = left.andedBy(right)
        elif node.opTok.matches(TT_KEYWORD, 'or'):
            result, error = left.oredBy(right)

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
        elif node.opTok.matches(TT_KEYWORD, 'not'):
            number, error = number.notted()

        if error:
            return res.failure(error)
        else:
            return res.success(number.setPos(node.posStart, node.posEnd))

    def visitIfNode(self, node, context):
        res = RTResult()

        for condition, expr in node.cases:
            conditionValue = res.register(self.visit(condition, context))
            if res.error:
                return res

            if conditionValue.isTrue():
                exprValue = res.register(self.visit(expr, context))
                if res.error:
                    return res
                return res.success(exprValue)

        if node.elseCase:
            elseValue = res.register(self.visit(node.elseCase, context))
            if res.error:
                return res
            return res.success(elseValue)

        return res.success(None)

    def visitForNode(self, node, context):
        res = RTResult()

        startValue = res.register(self.visit(node.startValueNode, context))
        if res.error:
            return res

        endValue = res.register(self.visit(node.endValueNode, context))
        if res.error:
            return res

        if node.stepValueNode:
            stepValue = res.register(self.visit(node.stepValueNode, context))
            if res.error:
                return res
        else:
            stepValue = Number(1)

        i = startValue.value

        if stepValue.value >= 0:
            condition = lambda: i < endValue.value
        else:
            condition = lambda: i > endValue.value

        while condition():
            context.symbolTable.set(node.varNameTok.value, Number(i))
            i += stepValue.value

            res.register(self.visit(node.bodyNode, context))
            if res.error:
                return res
# 6:15
        return res.success(None)


globalSymbolTable = SymbolTable()
globalSymbolTable.set("null", Number(0))
globalSymbolTable.set("true", Number(1))
globalSymbolTable.set("false", Number(0))


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
