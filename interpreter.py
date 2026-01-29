from enum import IntEnum, Enum, auto

class Keyword(IntEnum):
    LET = auto()
    CALL = auto()
    IF = auto()
    THEN = auto()
    ELSE = auto()
    FI = auto()
    WHILE = auto()
    DO = auto()
    OD = auto()
    RETURN = auto()
    VAR = auto()
    VOID = auto()
    FUNCTION = auto()
    MAIN = auto()

class State(IntEnum):
    EMPTY = auto()
    IDENTIFIER = auto()
    NUMBER = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()

class Keyword(str, Enum):
    LET = "let"
    CALL = "call"
    IF = "if"
    THEN = "then"
    ELSE = "else"
    FI = "fi"
    WHILE = "while"
    DO = "do"
    OD = "od"
    RETURN = "return"
    VAR = "var"
    VOID = "void"
    FUNCTION = "function"
    MAIN = "main"

class Token(str, Enum):
    ASSIGNMENT = "<-"
    LT = "<"
    LEQ = "<="
    GT = ">"
    GEQ = ">="
    NEQ = "!="
    EQ = "=="
    PLUS = "+"
    MINUS = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    SEPARATOR = ";"
    COMMA = ","
    OPEN_PARENTHESES = "("
    CLOSE_PARENTHESES = ")"
    OPEN_BRACE = "{"
    CLOSE_BRACE = "}"

class Identifier:
    def __init__(self):
        self.keywords = {keyword.value for keyword in Token}
        self.variables = dict()
        self.predefined_functions = {"InputNum", "OutputNum", "OutputNewLine"}
        self.functions = self.predefined_functions.copy()

class FileStream:
    def __init__(self, file_name: str):
        self.file = open(file_name, "r")
        self.token = None
    
    def peek_char(self):
        pos = self.file.tell()
        char = self.file.read(1)
        self.file.seek(pos)
        return char
    
    def next_char(self):
        return self.file.read(1)
    
    def peek_token(self):
        return self.token
        
    def next_token(self):
        buffer = []
        state = State.EMPTY
        
        while self.peek_char().isspace():
            self.next_char()

        while True:
            buffer.append(self.file.read(1))
            if buffer == ['2', '0', '1', '8']:
                print(buffer)
            elif buffer == ["f", "i"]:
                print(buffer)
            match state:
                case State.EMPTY:
                    if buffer[-1].isalpha():
                        state = State.IDENTIFIER
                    elif buffer[-1].isdigit():
                        state = State.NUMBER
                    elif buffer[-1] == "<":
                        state = State.LESS_THAN
                    elif buffer[-1] == ";":
                        break
                    elif buffer[-1] == ".":
                        break
                    elif buffer[-1] == ",":
                        break
                    elif buffer[-1] == "+":
                        break
                    elif buffer[-1] == "-":
                        break
                    elif buffer[-1] == "*":
                        break
                    elif buffer[-1] == "/":
                        break
                    elif buffer[-1] == "(":
                        break
                    elif buffer[-1] == ")":
                        break
                    elif buffer[-1] == "{":
                        break
                    elif buffer[-1] == "}":
                        break
                    elif buffer[-1] == "=":
                        state = State.EQUAL
                    elif buffer[-1] == "!":
                        state = State.NOT_EQUAL
                    elif buffer[-1] == ">":
                        state = State.GREATER_THAN
                    elif buffer[-1] == "":
                        break
                    else:
                        raise SyntaxError("You missed something")
                case State.IDENTIFIER:
                    if not buffer[-1].isalnum():
                        buffer.pop()
                        self.file.seek(self.file.tell()-1)
                        break
                case State.NUMBER:
                    if not buffer[-1].isnumeric():
                        buffer.pop()
                        self.file.seek(self.file.tell()-1)
                        break
                case State.EQUAL:
                    if buffer[-1] == "=":
                        break
                case State.LESS_THAN:
                    if buffer[-1] == "-":
                        break
                    elif buffer[-1] == "=":
                        break
                    elif buffer[-1] == " ":
                        buffer.pop()
                        break
                    else:
                        break
                case State.NOT_EQUAL:
                    if buffer[-1] == "=":
                        break
                case State.GREATER_THAN:
                    if buffer[-1] == "=":
                        break
                    else:
                        break
            
        self.token = "".join(buffer)
        if self.token == "2018":
            print(2018)
        return
            
    def close(self):
        self.file.close()


def emit(string: str) -> None:
    print(string)
    with open("interpreter_output.txt", "a") as file:
        file.write(f"{string}\n")
        

def interpret(file_name: str) -> None:
    file_stream = FileStream(file_name)
    identifier = Identifier()
    
    def consume_whitespace() -> None:
        while file_stream.peek_char().isspace():
            file_stream.next_char()
        return


    ## EXPRESSION PARSING
    def expression() -> int:
        result = term()
        while file_stream.peek_token() in ["+", "-"]:
            token = file_stream.peek_token()
            file_stream.next_token()
            match token:
                case "+":
                    result += term()
                case "-":
                    result -= term()
        return result

    def term() -> int:
        result = factor()
        while file_stream.peek_token() in ["*", "/"]:
            token = file_stream.peek_token()
            file_stream.next_token()
            match token:
                case "*":
                    result *= factor()
                case "/":
                    result /= factor()
        return result
    
    def factor() -> int:
        result = None
        if file_stream.peek_token() == "(":
            file_stream.next_token()
            result = expression()
            if file_stream.peek_token() == ")":
                file_stream.next_token()
            else:
                raise SyntaxError(f"Factor: no )")
        elif file_stream.peek_token().isalnum():
            result = number()
        else:
            raise SyntaxError(f"Factor no digits")
        return result

    def number() -> int:
        number = file_stream.peek_token()
        if not number.isdigit():
            number = identifier.variables[number]
        file_stream.next_token()
        return int(number)
    
    ## STATEMENT PARSING
    def assignment() -> None:
        file_stream.next_token() # consume Keyword.LET
        variable_name = file_stream.peek_token()
        file_stream.next_token() # consume variable_name
        if file_stream.peek_token() != Token.ASSIGNMENT:
            raise SyntaxError(f"Expected \'<-\' following \'let\' + identifier \'{variable_name}\'")
        file_stream.next_token() # consume Token.ASSIGNMENT
        identifier.variables[variable_name] = expression()
    
    def function_call() -> None:
        file_stream.next_token() # consume Keyword.CALL
        function_name = file_stream.peek_token()
        file_stream.next_token() # consume function_name
        function_arguments = []
        file_stream.next_token()
        if file_stream.peek_token() == Token.OPEN_PARENTHESES:
            file_stream.next_token() # consume TOken.OPEN_PARENTHESES
            while file_stream.peek_token() == Token.COMMA:
                function_arguments.append(expression())
            if file_stream.peek_token() != Token.CLOSE_PARENTHESES:
                raise SyntaxError(f"Expected \')\' to close called function arguments")
            file_stream.next_token()

    def if_then_else_fi() -> None:
        if file_stream.peek_token() != Keyword.IF:
            raise SyntaxError(f"Expected \'if'\ token")
        file_stream.next_token() # consume Keyword.IF
        left_hand_side = expression()
        relation_operator = file_stream.peek_token()
        file_stream.next_token()
        right_hand_side = expression()
        if file_stream.peek_token() != Keyword.THEN:
            raise SyntaxError(f"Expected \'then'\ token")
        file_stream.next_token() # consume Keyword.THEN
        statement_sequence()
        if file_stream.peek_token() == Keyword.ELSE:
            file_stream.next_token()
            statement_sequence()
        if file_stream.peek_token() != Keyword.FI:
            raise SyntaxError(f"Expected \'fi\' token")
        file_stream.next_token()
        
    def while_do_od() -> None:
        if file_stream.peek_token() != Keyword.WHILE:
            raise SyntaxError(f"Expected \'while\' token")
        file_stream.next_token() # consume Keyword.WHILE
        left_hand_side = expression()
        relation_operator = file_stream.peek_token()
        file_stream.next_token()
        right_hand_side = expression()
        if file_stream.peek_token() != Keyword.DO:
            raise SyntaxError(f"Expected \'do\' token")
        file_stream.next_token() # consume Keyword.DO
        statement()
        if file_stream.peek_token() != Keyword.OD:
            raise SyntaxError(f"Expected \'od\' token")
        file_stream.next_token()
    
    def returning() -> None:
        if file_stream.peek_token() != Keyword.RETURN:
            raise SyntaxError(f"Expected \'return\' token")
        file_stream.next_token() # consume Keyword.RETURN
        if file_stream.peek_token() != Token.SEPARATOR:
            result = expression()
        return

    def statement_sequence() -> None:
        #//FLAG PARSE STATEMENTS TO ACTUALLY DO SOMETHING
        statement()
        while file_stream.peek_token() == Token.SEPARATOR:
            file_stream.next_token()
            if file_stream.peek_token() == Token.CLOSE_BRACE:
                break
            statement()

    def statement() -> None:
        while True: # do-while loop for parsing statements
            match file_stream.peek_token():
                case Keyword.LET:
                    assignment()
                case Keyword.IF:
                    if_then_else_fi()
                case Keyword.ELSE:
                    break
                case Keyword.FI:
                    break
                case Keyword.WHILE:
                    while_do_od()
                case Keyword.RETURN:
                    returning()
                case Token.SEPARATOR:
                    file_stream.next_token()
                case Keyword.CALL:
                    function_call()
                case Keyword.VOID | Keyword.FUNCTION:
                    function_declaration()
                case _:
                    break
        
    def variable_declaration() -> list:
        variable_names = []
        if file_stream.peek_token() == Keyword.VAR:
            file_stream.next_token() # consume Keyword.VAR
            variable_names.append(file_stream.peek_token()) # var_name
            file_stream.next_token()
            while file_stream.peek_token() == Token.COMMA: # , or ;
                file_stream.next_token()
                variable_names.append(file_stream.peek_token()) # var_name
                file_stream.next_token()
            if file_stream.peek_token() != Token.SEPARATOR:
                raise SyntaxError(f"Expected \';\' to end variable declaration")
            file_stream.next_token()
        return variable_names

    def function_body() -> None:
        variable_names = variable_declaration()
        if file_stream.peek_token() != Token.OPEN_BRACE:
            raise SyntaxError(f"Expected \'{{\' to start statement sequence")
        file_stream.next_token()
        statement_sequence()
        if file_stream.peek_token() != Token.CLOSE_BRACE:
            raise SyntaxError(f"Expected \'}}\' to end statement sequence")
        file_stream.next_token()

    def formal_parameters() -> list:
        if file_stream.peek_token() != Token.OPEN_PARENTHESES:
            raise SyntaxError(f"Expected \'(\' to start formal params")
        file_stream.next_token()
        parameters = [file_stream.peek_token()]
        file_stream.next_token()
        while file_stream.peek_token() == Token.COMMA:
            file_stream.next_token()
            parameters.append(file_stream.peek_token())
            file_stream.next_token()
        if file_stream.peek_token() != Token.CLOSE_PARENTHESES:
            raise SyntaxError(f"Expected \')\' to close formal params")
        file_stream.next_token()
        return parameters

    def function_declaration() -> None:
        if file_stream.peek_token() == Keyword.VOID:
            file_stream.next_token()
        file_stream.next_token() # consume Keyword.FUNCTION
        function_name = file_stream.peek_token()
        file_stream.next_token() # consume function_name
        parameters = formal_parameters()
        file_stream.next_token() # consume ;
        function_body()
        if file_stream.peek_token() != Token.SEPARATOR:
            raise SyntaxError(f"Expected \';\' to end function body")
        file_stream.next_token()
        

    def computation() -> None:
        if file_stream.peek_token() == Keyword.MAIN:
            file_stream.next_token()
            variables = variable_declaration()
            while file_stream.peek_token() in [Keyword.VOID, Keyword.FUNCTION]:
                function_declaration()
            if file_stream.peek_token() != Token.OPEN_BRACE:
                raise SyntaxError(f"Expected \'{{\' to open statement sequence")
            file_stream.next_token()
            statement_sequence()
            if file_stream.peek_token() != Token.CLOSE_BRACE:
                raise SyntaxError(f"Expected \'}}\' to close statement sequence")
            file_stream.next_token()
        if file_stream.peek_token() != ".":
            raise SyntaxError(f"Expected \'.\' to end computation")
    
    tokens = []
    file_stream.next_token()
    while file_stream.peek_token() != "":
        if file_stream.peek_token() == "2018":
            print(1)
        tokens.append(file_stream.peek_token())
        file_stream.next_token()
    print(tokens)

    # file_stream.next_token()
    # computation()

    file_stream.close()


if __name__ == "__main__":
    from pathlib import Path
    interpret(Path(__file__).resolve().parent / "interpretee.txt")
