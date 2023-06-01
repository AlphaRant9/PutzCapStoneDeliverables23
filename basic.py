import string
import os
import math

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
TT_STRING = 'STRING'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MUL = 'MUL'
TT_DIV = 'DIV'
TT_POW = 'POW'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_LSQUARE = 'LSQUARE'
TT_RSQUARE = 'RSQUARE'
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
TT_COMMA = 'COMMA'
TT_ARROW = 'ARROW'
TT_NEWLINE = 'NEWLINE'

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
    'while',
    'fun',
    'end'

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
            elif self.currentChar in ';\n':
                tokens.append(Token(TT_NEWLINE, self.pos))
                self.advance()
            elif self.currentChar in DIGITS:
                tokens.append(self.makeNumber())
            elif self.currentChar.lower() in LETTERS:
                tokens.append(self.makeIdentifier())
            elif self.currentChar == '"':
                tokens.append(self.makeString())
            elif self.currentChar == '+':
                tokens.append(Token(TT_PLUS, posStart=self.pos))
                self.advance()
            elif self.currentChar == '-':
                tokens.append(self.makeMinusOrArrow())
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
            elif self.currentChar == '[':
                tokens.append(Token(TT_LSQUARE, posStart=self.pos))
                self.advance()
            elif self.currentChar == ']':
                tokens.append(Token(TT_RSQUARE, posStart=self.pos))
                self.advance()
            elif self.currentChar == ',':
                tokens.append(Token(TT_COMMA, posStart=self.pos))
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

    def makeString(self):
        string = ''
        posStart = self.pos.copy()
        escapeCharacter = False
        self.advance()

        escapeCharacters = {
            'n': '\n',
            't': '\t'
        }

        while self.currentChar != None and (self.currentChar != '"' or escapeCharacter):
            if escapeCharacter:
                string += escapeCharacters.get(self.currentChar, self.currentChar)
                escapeCharacter = False
            else:
                if self.currentChar == '\\':
                    escapeCharacter = True
                else:
                    string += self.currentChar
            self.advance()

        self.advance()
        return Token(TT_STRING, string, posStart, self.pos)

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

    def makeMinusOrArrow(self):
        tokType = TT_MINUS
        posStart = self.pos.copy()
        self.advance()

        if self.currentChar == '>':
            self.advance()
            tokType = TT_ARROW

        return Token(tokType, posStart=posStart, posEnd=self.pos)

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


class StringNode:
    def __init__(self, tok):
        self.tok = tok

        self.posStart = self.tok.posStart
        self.posEnd = self.tok.posEnd

    def __repr__(self):
        return f'{self.tok}'


class ListNode:
    def __init__(self, elementNodes, posStart, posEnd):
        self.elementNodes = elementNodes

        self.posStart = posStart
        self.posEnd = posEnd


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


class FuncDefNode:
    def __init__(self, varNameTok, argNameToks, bodyNode):
        self.varNameTok = varNameTok
        self.argNameToks = argNameToks
        self.bodyNode = bodyNode

        if self.varNameTok:
            self.posStart = self.varNameTok.posStart
        elif len(self.argNameToks) > 0:
            self.posStart = self.argNameToks[0].posStart
        else:
            self.posStart = self.bodyNode.posStart

        self.posEnd = self.bodyNode.posEnd


class CallNode:
    def __init__(self, nodeToCall, argNodes):
        self.nodeToCall = nodeToCall
        self.argNodes = argNodes

        self.posStart = self.nodeToCall.posStart

        if len(self.argNodes) > 0:
            self.posEnd = self.argNodes[len(self.argNodes) - 1].posEnd
        else:
            self.posEnd = self.nodeToCall.posEnd


class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advanceCount = 0
        self.toReverseCount = 0

    def registerAdvancement(self):
        self.advanceCount += 1

    def register(self, res):
        self.advanceCount += res.advanceCount
        if res.error:
            self.error = res.error
        return res.node

    def tryRegister(self, res):
        if res.error:
            self.toReverseCount = res.advanceCount
            return None
        return self.register(res)

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

    def advance(self):
        self.tokIdx += 1
        if self.tokIdx < len(self.tokens):
            self.currentTok = self.tokens[self.tokIdx]
        return self.currentTok

    def reverse(self, amount=1):
        self.tokIdx -= amount
        self.updateCurrentTok()
        return self.currentTok

    def updateCurrentTok(self):
        if 0 <= self.tokIdx < len(self.tokens):
            self.currentTok = self.tokens[self.tokIdx]

    def parse(self):
        res = self.statements()
        if not res.error and self.currentTok.type != TT_EOF:
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                "Expected '+', '-', '*' or '/'"
            ))
        return res

    def call(self):
        res = ParseResult()
        atom = res.register(self.atom())
        if res.error:
            return res

        if self.currentTok.type == TT_LPAREN:
            res.registerAdvancement()
            self.advance()

            argNodes = []
            if self.currentTok.type == TT_RPAREN:
                res.registerAdvancement()
                self.advance()
            else:
                argNodes.append(res.register(self.expr()))
                if res.error:
                    return res.failure(InvalidSyntaxError(
                        self.currentTok.posStart, self.currentTok.posEnd,
                        "Expected ')', 'var', 'if', 'for', 'while', 'fun', int, float, '[', identifier, '+', '-' or '('"
                    ))

                while self.currentTok.type == TT_COMMA:
                    res.registerAdvancement()
                    self.advance()

                    argNodes.append(res.register(self.expr()))
                    if res.error:
                        return res

                if self.currentTok.type != TT_RPAREN:
                    return res.failure(InvalidSyntaxError(
                        self.currentTok.posStart, self.currentTok.posEnd,
                        f"Expected ',' or ')'"
                    ))

                res.registerAdvancement()
                self.advance()

            return res.success(CallNode(atom, argNodes))
        return res.success(atom)


    def atom(self):
        res = ParseResult()
        tok = self.currentTok

        if tok.type in (TT_INT, TT_FLOAT):
            res.registerAdvancement()
            self.advance()
            return res.success(NumberNode(tok))

        elif tok.type == TT_STRING:
            res.registerAdvancement()
            self.advance()
            return res.success(StringNode(tok))

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

        elif tok.type == TT_LSQUARE:
            listExpr = res.register(self.listExpr())
            if res.error:
                return res
            return res.success(listExpr)

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
        elif tok.matches(TT_KEYWORD, 'fun'):
            funcDef = res.register(self.funcDef())
            if res.error:
                return res
            return res.success(funcDef)

        return res.failure(InvalidSyntaxError(
            tok.posStart, tok.posEnd,
            "Expected 'var', 'if', 'for', 'while', 'fun', '[', int, float, identifier, '+', '-' or '('"
        ))

    def listExpr(self):
        res = ParseResult()
        elementNodes = []
        posStart = self.currentTok.posStart.copy()

        if self.currentTok.type != TT_LSQUARE:
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"Expected '['"
            ))

        res.registerAdvancement()
        self.advance()

        if self.currentTok.type == TT_RSQUARE:
            res.registerAdvancement()
            self.advance()
        else:
            elementNodes.append(res.register(self.expr()))
            if res.error:
                return res.failure(InvalidSyntaxError(
                    self.currentTok.posStart, self.currentTok.posEnd,
                    "Expected ']', 'var', 'if', 'for', 'while', 'fun', int, float, '[', identifier, '+', '-' or '('"
                ))

            while self.currentTok.type == TT_COMMA:
                res.registerAdvancement()
                self.advance()

                elementNodes.append(res.register(self.expr()))
                if res.error:
                    return res

            if self.currentTok.type != TT_RSQUARE:
                return res.failure(InvalidSyntaxError(
                    self.currentTok.posStart, self.currentTok.posEnd,
                    f"Expected ',' or ']'"
                ))

            res.registerAdvancement()
            self.advance()

        return res.success(ListNode(
            elementNodes,
            posStart,
            self.currentTok.posEnd.copy()
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

    def whileExpr(self):
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

    def funcDef(self):
        res = ParseResult()

        if not self.currentTok.matches(TT_KEYWORD, 'fun'):
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"Expected 'fun'"
            ))

        res.registerAdvancement()
        self.advance()

        if self.currentTok.type == TT_IDENTIFIER:
            varNameTok = self.currentTok
            res.registerAdvancement()
            self.advance()
            if self.currentTok.type != TT_LPAREN:
                return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"Expected '('"
            ))
        else:
            varNameTok = None
            res.registerAdvancement()
            self.advance()
            if self.currentTok.type != TT_LPAREN:
                return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"Expected identifier or '('"
            ))

        res.registerAdvancement()
        self.advance()

        argNameToks = []
        if self.currentTok.type == TT_IDENTIFIER:
            argNameToks.append(self.currentTok)
            res.registerAdvancement()
            self.advance()

            while self.currentTok.type == TT_COMMA:
                res.registerAdvancement()
                self.advance()

                if self.currentTok.type != TT_IDENTIFIER:
                    return res.failure(InvalidSyntaxError(
                        self.currentTok.posStart, self.currentTok.posEnd,
                        f"Expected identifier"
                    ))

                argNameToks.append(self.currentTok)
                res.registerAdvancement()
                self.advance()

            if self.currentTok.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.currentTok.posStart, self.currentTok.posEnd,
                    f"Expected ',' or ')'"
                ))
        else:
            if self.currentTok.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"Expected identifier or ')'"
            ))

        res.registerAdvancement()
        self.advance()

        if self.currentTok.type != TT_ARROW:
            return res.failure(InvalidSyntaxError(
                self.currentTok.posStart, self.currentTok.posEnd,
                f"Expected '->'"
            ))

        res.registerAdvancement()
        self.advance()

        nodeToReturn = res.register(self.expr())
        if res.error:
            return res

        return res.success(FuncDefNode(varNameTok, argNameToks, nodeToReturn))

    def power(self):
        return self.binOp(self.call, (TT_POW,), self.factor)

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
                                                  "Expected int, float, '[', identifier, '+', '-', '(', or 'not'"))

        return res.success(node)

    def statements(self):
        res = ParseResult()
        statements = []
        posStart = self.currentTok.posStart.copy()

        while self.currentTok.type == TT_NEWLINE:
            res.registerAdvancement()
            self.advance()

        statement = res.register(self.expr())
        if res.error:
            return res
        statements.append(statement)

        moreStatements = True

        while True:
            newLineCount = 0
            while self.currentTok.type == TT_NEWLINE:
                res.registerAdvancement()
                self.advance()
                newLineCount += 1
            if newLineCount == 0:
                moreStatements = False

            if not moreStatements:
                break

            statement = res.tryRegister(self.expr())
            if not statement:
                self.reverse(res.toReverseCount)
                moreStatements = False
                continue

            statements.append(statement)

        return res.success(ListNode(
            statements,
            posStart,
            self.currentTok.posEnd.copy()
        ))

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
                    "Expected 'var', 'if', 'for', 'while', 'fun', int, float, '[', identifier, '+', '-', '(', or 'not'"
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


class Value:
    def __init__(self):
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
        return None, self.illegalOperation(other)

    def subbedBy(self, other):
        return None, self.illegalOperation(other)

    def multedBy(self, other):
        return None, self.illegalOperation(other)

    def divedBy(self, other):
        return None, self.illegalOperation(other)

    def powedBy(self, other):
        return None, self.illegalOperation(other)

    def getComparisonEq(self, other):
        return None, self.illegalOperation(other)

    def getComparisonNe(self, other):
        return None, self.illegalOperation(other)

    def getComparisonGt(self, other):
        return None, self.illegalOperation(other)

    def getComparisonGte(self, other):
        return None, self.illegalOperation(other)

    def getComparisonLt(self, other):
        return None, self.illegalOperation(other)

    def getComparisonLte(self, other):
        return None, self.illegalOperation(other)

    def andedBy(self, other):
        return None, self.illegalOperation(other)

    def oredBy(self, other):
        return None, self.illegalOperation(other)

    def notted(self, other):
        return None, self.illegalOperation(other)

    def execute(self, args):
        return None, self.illegalOperation()

    def copy(self):
        raise Exception('No copy method defined')

    def isTrue(self):
        return False

    def illegalOperation(self, other=None):
        if not other:
            other = self
        return RTError(
            self.posStart, other.posEnd,
            'Illegal Operation',
            self.context
        )


class Number(Value):
    def __init__(self, value):
        self.value = value
        self.setPos()
        self.setContext()

    def addedTo(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).setContext(self.context), None
        else:
            return None, Value.illegalOperation(self.posStart, other.posEnd)

    def subbedBy(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).setContext(self.context), None
        else:
            return None, Value.illegalOperation(self.posStart, other.posEnd)

    def multedBy(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).setContext(self.context), None
        else:
            return None, Value.illegalOperation(self.posStart, other.posEnd)

    def divedBy(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.posStart, other.posEnd,
                    'Division by zero',
                    self.context
                )

            return Number(self.value / other.value).setContext(self.context), None
        else:
            return None, Value.illegalOperation(self.posStart, other.posEnd)

    def powedBy(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).setContext(self.context), None
        else:
            return None, Value.illegalOperation(self.posStart, other.posEnd)

    def getComparisonEq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).setContext(self.context), None
        else:
            return None, Value.illegalOperation(self.posStart, other.posEnd)

    def getComparisonNe(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).setContext(self.context), None
        else:
            return None, Value.illegalOperation(self.posStart, other.posEnd)

    def getComparisonLt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).setContext(self.context), None
        else:
            return None, Value.illegalOperation(self.posStart, other.posEnd)

    def getComparisonGt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).setContext(self.context), None
        else:
            return None, Value.illegalOperation(self.posStart, other.posEnd)

    def getComparisonLte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).setContext(self.context), None
        else:
            return None, Value.illegalOperation(self.posStart, other.posEnd)

    def getComparisonGte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).setContext(self.context), None
        else:
            return None, Value.illegalOperation(self.posStart, other.posEnd)

    def andedBy(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).setContext(self.context), None
        else:
            return None, Value.illegalOperation(self.posStart, other.posEnd)

    def oredBy(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).setContext(self.context), None
        else:
            return None, Value.illegalOperation(self.posStart, other.posEnd)

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


Number.null = Number(0)
Number.true = Number(1)
Number.false = Number(0)
Number.mathPi = Number(math.pi)


class String(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def addedTo(self, other):
        if isinstance(other, String):
            return String(self.value + other.value).setContext(self.context), None
        else:
            return None, Value.illegalOperation(self, other)

    def multedBy(self, other):
        if isinstance(other, Number):
            return String(self.value * other.value).setContext(self.context), None
        else:
            return None, Value.illegalOperation(self, other)

    def isTrue(self):
        return len(self.value) > 0

    def copy(self):
        copy = String(self.value)
        copy.setPos(self.posStart, self.posEnd)
        copy.setContext(self.context)
        return copy

    def __str__(self):
        return self.value

    def __repr__(self):
        return f'"{self.value}"'


class List(Value):
    def __init__(self, elements):
        super().__init__()
        self.elements = elements

    def addedTo(self, other):
        newList = self.copy()
        newList.elements.append(other)
        return newList, None

    def subbedBy(self, other):
        if isinstance(other, Number):
            newList = self.copy()
            try:
                newList.elements.pop(other.value)
                return newList, None
            except:
                return None, RTError(
                    other.posStart, other.posEnd,
                    'Elements at this index could not be removed from list because index is out of bounds',
                    self.context
                )
        else:
            return None, Value.illegalOperation(self, other)


    def multedBy(self, other):
        if isinstance(other, List):
            newList = self.copy()
            newList.elements.extend(other.elements)
            return newList, None
        else:
            return None, Value.illegalOperation(self, other)

    def divedBy(self, other):
        if isinstance(other, Number):
            try:
                return self.elements[other.value], None
            except:
                return None, RTError(
                    other.posStart, other.posEnd,
                    'Elements at this index could not be retrieved from list because index is out of bounds',
                    self.context
                )
        else:
            return None, Value.illegalOperation(self, other)

    def copy(self):
        copy = List(self.elements)
        copy.setPos(self.posStart, self.posEnd)
        copy.setContext(self.context)
        return copy

    def __str__(self):
        return f'{", ".join([str(x) for x in self.elements])}'

    def __repr__(self):
        return f'[{", ".join([str(x) for x in self.elements])}]'


class BaseFunction(Value):
    def __init__(self, name):
        super().__init__()
        self.name = name or '<anonymous>'

    def generateNewContext(self):
        newContext = Context(self.name, self.context, self.posStart)
        newContext.symbolTable = SymbolTable(newContext.parent.symbolTable)
        return newContext

    def checkArgs(self, argNames, args):
        res = RTResult()
        if len(args) > len(argNames):
            return res.failure(RTError(
                self.posStart, self.posEnd,
                f"{len(args) - len(argNames)} too many args passed into '{self.name}'."
            ))

        if len(args) < len(argNames):
            return res.failure(RTError(
                self.posStart, self.posEnd,
                f"{len(argNames) - len(args)} too few args passed into '{self.name}'."
            ))

        return res.success(None)

    def populateArgs(self, argNames, args, execCtx):
        for i in range(len(args)):
            argName = argNames[i]
            argValue = args[i]
            argValue.setContext(execCtx)
            execCtx.symbolTable.set(argName, argValue)

    def checkAndPopulateArgs(self, argNames, args, execCtx):
        res = RTResult()
        res.register(self.checkArgs(argNames, args))
        if res.error:
            return res
        self.populateArgs(argNames, args, execCtx)
        return res.success(None)


class Function(BaseFunction):
    def __init__(self, name, bodyNode, argNames):
        super().__init__(name)
        self.bodyNode = bodyNode
        self.argNames = argNames

    def execute(self, args):
        res = RTResult()
        interpreter = Interpreter()

        execCtx = self.generateNewContext()

        res.register(self.checkAndPopulateArgs(self.argNames, args, execCtx))
        if res.error:
            return res

        value = res.register(interpreter.visit(self.bodyNode, execCtx))
        if res.error:
            return res

        return res.success(value)

    def copy(self):
        copy = Function(self.name, self.bodyNode, self.argNames)
        copy.setContext(self.context)
        copy.setPos(self.posStart, self.posEnd)
        return copy

    def __repr__(self):
        return f"<function {self.name}>"


class BuiltInFunction(BaseFunction):
    def __init__(self, name):
        super().__init__(name)

    def execute(self, args):
        res = RTResult()
        execCtx = self.generateNewContext()

        methodName = f'execute_{self.name}'
        method = getattr(self, methodName, self.noVisitMethod)

        res.register(self.checkAndPopulateArgs(method.argNames, args, execCtx))
        if res.error:
            return res

        returnValue = res.register(method(execCtx))
        if res.error:
            return res
        return res.success(returnValue)

    def noVisitMethod(self, node, context):
        raise Exception(f'No execute_{self.name} method defined')

    def copy(self):
        copy = BuiltInFunction(self.name)
        copy.setContext(self.context)
        copy.setPos(self.posStart, self.posEnd)
        return copy

    def __repr__(self):
        return f"<built-in function {self.name}>"

    def execute_print(self, execCtx):
        print(str(execCtx.symbolTable.get('value')))
        return RTResult().success(Number.null)
    execute_print.argNames = ["value"]

    def execute_printRet(self, execCtx):
        return RTResult().success(String(str(execCtx.symbolTable.get('value'))))
    execute_printRet.argNames = ["value"]

    def execute_input(self, execCtx):
        text = input()
        return RTResult().success(String(text))
    execute_input.argNames = []

    def execute_inputInt(self, execCtx):
        while True:
            text = input()
            try:
                number = int(text)
                break
            except ValueError:
                print(f"'{text}' must be an integer. Try again!")
        return RTResult().success(Number(number))
    execute_inputInt.argNames = []

    def execute_inputFloat(self, execCtx):
        while True:
            text = input()
            try:
                number = float(text)
                break
            except ValueError:
                print(f"'{text}' must be an number. Try again!")
        return RTResult().success(Number(number))
    execute_inputFloat.argNames = []

    def execute_clear(self, execCtx):
        os.system('cls' if os.name == 'nt' else 'clear')
        return RTResult().success(Number.null)
    execute_clear.argNames = []

    def execute_isNumber(self, execCtx):
        isNumber = isinstance(execCtx.symbolTable.get("value"), Number)
        if isNumber:
            return RTResult().success(Number.true)
        else:
            return RTResult().success(Number.false)
    execute_isNumber.argNames = ['value']

    def execute_isString(self, execCtx):
        isString = isinstance(execCtx.symbolTable.get("value"), String)
        return RTResult().success(Number.true if isString else Number.false)
    execute_isString.argNames = ['value']

    def execute_isList(self, execCtx):
        isList = isinstance(execCtx.symbolTable.get("value"), List)
        return RTResult().success(Number.true if isList else Number.false)
    execute_isList.argNames = ['value']

    def execute_isFunction(self, execCtx):
        isFunction = isinstance(execCtx.symbolTable.get("value"), BaseFunction)
        return RTResult().success(Number.true if isFunction else Number.false)
    execute_isFunction.argNames = ['value']

    def execute_append(self, execCtx):
        list_ = execCtx.symbolTable.get("list")
        value_ = execCtx.symbolTable.get("value")

        if not isinstance(list_, List):
            return RTResult().failure(RTError(
                self.posStart, self.posEnd,
                "First argument must be list",
                execCtx
            ))

        list_.elements.append(value_)
        return RTResult().success(Number.null)
    execute_append.argNames = ['list', 'value']

    def execute_pop(self, execCtx):

        list_ = execCtx.symbolTable.get("list")
        index_ = execCtx.symbolTable.get("index")

        if not isinstance(list_, List):
            return RTResult().failure(RTError(
                self.posStart, self.posEnd,
                "First argument must be list",
                execCtx
            ))

        if not isinstance(index_, Number):
            return RTResult().failure(RTError(
                self.posStart, self.posEnd,
                "Second argument must be a number",
                execCtx
            ))

        try:
            element = list_.elements.pop(index_.value)
        except:
            return RTResult().failure(RTError(
                self.posStart, self.posEnd,
                'Element at this index could not be removed from list because index is out of range',
                execCtx
            ))

        return RTResult().success(element)
    execute_pop.argNames = ['list', 'index']

    def execute_extend(self, execCtx):
        listA = execCtx.symbolTable.get("listA")
        listB = execCtx.symbolTable.get("listB")

        if not isinstance(listA, List):
            return RTResult().failure(RTError(
                self.posStart, self.posEnd,
                "First argument must be list",
                execCtx
            ))

        if not isinstance(listB, List):
            return RTResult().failure(RTError(
                self.posStart, self.posEnd,
                "Second argument must be list",
                execCtx
            ))

        listA.elements.extend(listB.elements)
        return RTResult().success(Number.null)
    execute_pop.argNames = ['listA', 'listB']


BuiltInFunction.print       = BuiltInFunction("print")
BuiltInFunction.printRet    = BuiltInFunction("printRet")
BuiltInFunction.input       = BuiltInFunction("input")
BuiltInFunction.inputInt    = BuiltInFunction("inputInt")
BuiltInFunction.inputFloat  = BuiltInFunction("inputFloat")
BuiltInFunction.clear       = BuiltInFunction("clear")
BuiltInFunction.isNumber    = BuiltInFunction("isNumber")
BuiltInFunction.isString    = BuiltInFunction("isString")
BuiltInFunction.isList      = BuiltInFunction("isList")
BuiltInFunction.isFunction  = BuiltInFunction("isFunction")
BuiltInFunction.append      = BuiltInFunction("append")
BuiltInFunction.pop         = BuiltInFunction("pop")
BuiltInFunction.extend      = BuiltInFunction("extend")


class Context:
    def __init__(self, displayName, parent=None, parentEntryPos=None):
        self.displayName = displayName
        self.parent = parent
        self.parentEntryPos = parentEntryPos
        self.symbolTable = None


class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

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

    def visitStringNode(self, node, context):
        return RTResult().success(
            String(node.tok.value).setContext(context).setPos(node.posStart, node.posEnd)
        )

    def visitListNode(self, node, context):
        res = RTResult()
        elements = []

        for elementNode in node.elementNodes:
            elements.append(res.register(self.visit(elementNode, context)))
            if res.error:
                return res

        return res.success(
            List(elements).setContext(context).setPos(node.posStart, node.posEnd)
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

        value = value.copy().setPos(node.posStart, node.posEnd).setContext(context)
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
        elements = []

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

            elements.append(res.register(self.visit(node.bodyNode, context)))
            if res.error:
                return res

        return res.success(
            List(elements).setContext(context).setPos(node.posStart, node.posEnd)
        )

    def visitWhileNode(self, node, context):
        res = RTResult()
        elements = []

        while True:
            condition = res.register(self.visit(node.conditionNode, context))
            if res.error:
                return res

            if not condition.isTrue():
                break

            elements.append(res.register(self.visit(node.bodyNode, context)))
            if res.error:
                return res

        return res.success(
            List(elements).setContext(context).setPos(node.posStart, node.posEnd)
        )

    def visitFuncDefNode(self, node, context):
        res = RTResult()

        funcName = node.varNameTok.value if node.varNameTok else None
        bodyNode = node.bodyNode
        argNames = [argName.value for argName in node.argNameToks]
        funcValue = Function(funcName, bodyNode, argNames).setContext(context).setPos(node.posStart, node.posEnd)

        if node.varNameTok:
            context.symbolTable.set(funcName, funcValue)

        return res.success(funcValue)

    def visitCallNode(self, node, context):
        res = RTResult()
        args = []

        valueToCall = res.register(self.visit(node.nodeToCall, context))
        if res.error:
            return res
        valueToCall = valueToCall.copy().setPos(node.posStart, node.posEnd)

        for argNode in node.argNodes:
            args.append(res.register(self.visit(argNode, context)))
            if res.error:
                return res

        returnValue = res.register(valueToCall.execute(args))
        if res.error:
            return res

        returnValue = returnValue.copy().setPos(node.posStart, node.posEnd).setContext(context)
        return res.success(returnValue)


globalSymbolTable = SymbolTable()
globalSymbolTable.set("null", Number.null)
globalSymbolTable.set("true", Number.true)
globalSymbolTable.set("false", Number.false)
globalSymbolTable.set("pi", Number.mathPi)
globalSymbolTable.set("print", BuiltInFunction.print)
globalSymbolTable.set("printRet", BuiltInFunction.printRet)
globalSymbolTable.set("input", BuiltInFunction.input)
globalSymbolTable.set("inputInt", BuiltInFunction.inputInt)
globalSymbolTable.set("inputFloat", BuiltInFunction.inputFloat)
globalSymbolTable.set("clear", BuiltInFunction.clear)
globalSymbolTable.set("cls", BuiltInFunction.clear)
globalSymbolTable.set("isNum", BuiltInFunction.isNumber)
globalSymbolTable.set("isStr", BuiltInFunction.isString)
globalSymbolTable.set("isList", BuiltInFunction.isList)
globalSymbolTable.set("isFun", BuiltInFunction.isNumber)
globalSymbolTable.set("append", BuiltInFunction.append)
globalSymbolTable.set("pop", BuiltInFunction.pop)
globalSymbolTable.set("extend", BuiltInFunction.extend)


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
