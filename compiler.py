from tokenizer import Keyword, Token, SourceCode

class ThingThatTracksUserDefinedStuffAndOtherThings:
    # This class' features will be integrated into another class later
    def __init__(self):
        self.keywords = {keyword.value for keyword in Token}
        self.variables = dict()
        self.predefined_functions = {"InputNum", "OutputNum", "OutputNewLine"}
        self.functions = self.predefined_functions.copy()


def emit(string: str) -> None:
    # Not used currently
    print(string)
    with open("interpreter_output.txt", "a") as file:
        file.write(f"{string}\n")
        

def compile(file_name: str) -> None:
    file_stream = SourceCode(file_name)
    my_thing_be_tracking_YO = ThingThatTracksUserDefinedStuffAndOtherThings()
    
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
            number = my_thing_be_tracking_YO.variables[number]
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
        my_thing_be_tracking_YO.variables[variable_name] = expression()
    
    def function_call() -> None:
        file_stream.next_token() # consume Keyword.CALL
        function_name = file_stream.peek_token()
        file_stream.next_token() # consume function_name
        function_arguments = []
        if file_stream.peek_token() == Token.OPEN_PARENTHESES:
            file_stream.next_token() # consume Token.OPEN_PARENTHESES
            function_arguments.append(file_stream.peek_token())
            file_stream.next_token() # consume first function argument
            while file_stream.peek_token() == Token.COMMA:
                function_arguments.append(expression())
            if file_stream.peek_token() != Token.CLOSE_PARENTHESES:
                raise SyntaxError(f"Expected \')\' to close called function arguments")
            file_stream.next_token()

    def if_then_else_fi() -> None:
        if file_stream.peek_token() != Keyword.IF:
            raise SyntaxError(f"Expected \'if\' token")
        file_stream.next_token() # consume Keyword.IF
        left_hand_side = expression()
        relation_operator = file_stream.peek_token()
        file_stream.next_token()
        right_hand_side = expression()
        if file_stream.peek_token() != Keyword.THEN:
            raise SyntaxError(f"Expected \'then\' token")
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
        statement_sequence()
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
                case Keyword.OD:
                    break
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
        tokens.append(file_stream.peek_token())
        file_stream.next_token()
    print(tokens)

    # actual program execution
    file_stream = SourceCode(file_name)
    file_stream.next_token()
    computation()


if __name__ == "__main__":
    from pathlib import Path
    compile(Path(__file__).resolve().parent / "interpretee.txt")
