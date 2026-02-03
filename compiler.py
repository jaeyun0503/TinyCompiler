from sys import stdout as STDOUT
from enum import Enum
from pathlib import Path

from tokenizer import Keyword, Token, SourceCode
from output import OutStream, open_files, close_files

# MODIFY TO TARGET SOURCE CODE FILE
SOURCE_CODE_FILE_NAME = "source_code_simple_math.txt"

# MODIFY TO CHANGE OUTPUT STREAMS
OUTPUT_TO_CONSOLE = True
ADDITIONAL_OUTPUT_FILE_NAMES = []

class Operator(Enum):
    CONST = "const #"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    CMP = "cmp"
    PHI = "phi"
    END = "end"
    BRANCH = "bra"
    BRANCH_NOT_EQUAL = "bne"
    BRANCH_EQUAL = "beq"
    BRANCH_LESS_THAN_EQUAL = "ble"
    BRANCH_LESS_THAN = "blt"
    BRANCH_GREATER_THAN_EQUAL = "bge"
    BRANCH_GREATER_THAN = "bgt"
    READ = "read"
    WRITE = "write"
    WRITE_NEW_LINE = "writeNL"
    JUMP_SUBROUTINE = "jsr"
    RETURN = "return"
    GET_PARAMETER_1 = "getpar1"
    GET_PARAMETER_2 = "getpar2"
    GET_PARAMETER_3 = "getpar3"
    SET_PARAMETER_1 = "setpar1"

    def __str__(self) -> str:
        return self.value
    

class IntermediateRepresentation:
    def __init__(self):
        self.basic_blocks = {0: []}
        self._basic_block_number = 1
        self._ssa_value = 1
        self._ssa_const_value = -1
        self.ssa_associations = dict()

        self.variables = dict()
        self._predefined_functions = {"InputNum", "OutputNum", "OutputNewLine"}
        self.functions = self._predefined_functions.copy()
        self.constants = set()

    def fetch_next_ssa_value(self) -> int:
        """
        Returns next available SSA value and
        prepares next available SSA value
        """
        self._ssa_value += 1
        return self._ssa_value - 1
    
    def fetch_next_ssa_const_value(self) -> int:
        """
        Returns next avaiable SSA value for constants and
        prepares next available SSA value for constants
        """
        self._ssa_const_value -= 1
        return self._ssa_const_value + 1
    
    def fetch_next_basic_block_number(self) -> int:
        self._basic_block_number += 1
        return self._basic_block_number - 1
    
    def get_ssa_value(self, operand: int | str) -> int:
        if operand in self.ssa_associations:
            return self.ssa_associations[operand]
        else: # should technically only be constants
            ssa_value = self.fetch_next_ssa_const_value()
            self.ssa_associations[operand] = ssa_value
            instruction = self.compose_instruction(ssa_value, Operator.CONST, operand)
            self.basic_blocks[0].append(instruction)
            emit(instruction)
            return ssa_value

    def map_operation(self, operation: str) -> str:
        # unused
        return

    def compose_instruction(self, line_number: int, operator: str, *operands) -> str:
        x, y, *_ = (*operands, "", "")
        space = "" if operator == Operator.CONST else " "
        return f"{line_number}: {operator}{space}{x} {y}".rstrip()
        
    def emit_instruction(self, instruction: str) -> None:
        # unused
        print(instruction)
        # write to object file

def emit(content: str = "") -> None:
    print(content, file=outstream)

outstream = None
def compile(file_name: str = None) -> None:
    # Data structures for parsing and IR generation
    file_stream = SourceCode(Path(__file__).resolve().parent / Path(SOURCE_CODE_FILE_NAME))
    intrepr = IntermediateRepresentation()

    # Output handling
    OBJECT_FILE_PATH = Path(__file__).resolve().parent / Path(SOURCE_CODE_FILE_NAME).with_suffix(".o")
    output_file_paths = [OBJECT_FILE_PATH]
    output_file_paths.extend([Path(__file__).resolve().parent / file_name for file_name in ADDITIONAL_OUTPUT_FILE_NAMES])
    output_streams = open_files(output_file_paths)
    output_streams.append(STDOUT) if OUTPUT_TO_CONSOLE else None
    global outstream
    outstream = OutStream(*output_streams)

    ## EXPRESSION PARSING
    def expression() -> int:
        ssa_value = term()
        while file_stream.peek_token() in ["+", "-"]:
            token = file_stream.peek_token()
            file_stream.next_token()
            new_ssa_value = intrepr.fetch_next_ssa_value()
            match token:
                case Token.PLUS:
                    emit(intrepr.compose_instruction(new_ssa_value, Operator.ADD, ssa_value, term()))
                case Token.MINUS:
                    emit(intrepr.compose_instruction(new_ssa_value, Operator.SUB, ssa_value, term()))
            ssa_value = new_ssa_value
        return ssa_value

    def term() -> int:
        ssa_value = factor()
        while file_stream.peek_token() in ["*", "/"]:
            token = file_stream.peek_token()
            file_stream.next_token()
            new_ssa_value = intrepr.fetch_next_ssa_value()
            match token:
                case Token.MULTIPLY:
                    emit(intrepr.compose_instruction(new_ssa_value, Operator.MUL, ssa_value, factor()))
                case Token.DIVIDE:
                    emit(intrepr.compose_instruction(new_ssa_value, Operator.DIV, ssa_value, factor()))
            ssa_value = new_ssa_value
        return ssa_value
    
    def factor() -> int:
        ssa_value = None
        if file_stream.peek_token() == "(":
            file_stream.next_token()
            ssa_value = expression()
            if file_stream.peek_token() == ")":
                file_stream.next_token()
            else:
                raise SyntaxError(f"Factor: no )")
        elif file_stream.peek_token().isalnum():
            ssa_value = number()
        else:
            raise SyntaxError(f"Factor no digits")
        return ssa_value

    def number() -> int:
        operand = file_stream.peek_token()
        file_stream.next_token()
        return intrepr.get_ssa_value(operand)
    
    ## STATEMENT PARSING
    def assignment() -> None:
        file_stream.next_token() # consume Keyword.LET
        variable_name = file_stream.peek_token()
        file_stream.next_token() # consume variable_name
        if file_stream.peek_token() != Token.ASSIGNMENT:
            raise SyntaxError(f"Expected \'<-\' following \'let\' + identifier \'{variable_name}\'")
        file_stream.next_token() # consume Token.ASSIGNMENT
        intrepr.ssa_associations[variable_name] = expression() # map variable_name to SSA value
    
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
            variable_names.append(file_stream.peek_token()) # save var_name
            file_stream.next_token()
            while file_stream.peek_token() == Token.COMMA: # , or ;
                file_stream.next_token()
                variable_names.append(file_stream.peek_token()) # save var_name
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
        """
        Parses entire source code
        """
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
    
    # tokens = []
    # file_stream.next_token()
    # while file_stream.peek_token() != "":
    #     tokens.append(file_stream.peek_token())
    #     file_stream.next_token()
    # print(tokens)

    # actual program execution
    file_stream.next_token()
    computation()
    emit()

    # closing all output streams except STDOUT
    close_files(*output_streams[1:])
    

if __name__ == "__main__":
    compile()