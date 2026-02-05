from sys import stdout as STDOUT
from enum import Enum
from pathlib import Path

from tokenizer import Keyword, Token, SourceCode
from output import OutStream, open_files, close_files

# MODIFY TO TARGET SOURCE CODE FILE
SOURCE_CODE_FILE_NAME = "source_code_if_else.txt"

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
    
    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def map_comparison_operator(cls, token: str):
        match token:
            case Token.LESS_THAN:
                return Operator.BRANCH_GREATER_THAN_EQUAL
            case Token.LESS_THAN_EQUAL:
                return Operator.BRANCH_GREATER_THAN
            case Token.GREATER_THAN:
                return Operator.BRANCH_LESS_THAN_EQUAL
            case Token.GREATER_THAN_EQUAL:
                return Operator.BRANCH_LESS_THAN
            case Token.EQUAL:
                return Operator.BRANCH_NOT_EQUAL
            case Token.NOT_EQUAL:
                return Operator.BRANCH_EQUAL
            case _:
                raise SyntaxError(f"Expected valid comparison operator token")

class Instruction():
    def __init__(self, ssa_value: int = None, operator: Operator = None, operand_1: str = "", operand_2: str = ""):
        self.ssa_value = ssa_value
        self.operator = operator
        self.operand_1 = operand_1
        self.operand_2 = operand_2

    def __str__(self) -> str:
        space = "" if self.operator == Operator.CONST else " "
        return f"{self.ssa_value}: {self.operator}{space}{self.operand_1} {self.operand_2}".rstrip()
    
    def __repr__(self) -> str:
        return self.__str__()

class BasicBlock():
    def __init__(self, block_number: int = None):
        self.number = block_number
        self.instructions = []
        self.join_instructions = []
        self.parent_blocks = set()
        self.variables_to_most_recent_ssa_value = dict()
    
    def get_starting_ssa_value(self) -> int:
        if self.join_instructions:
            return self.join_instructions[0].ssa_value
        elif self.instructions:
            return self.instructions[0].ssa_value
        else:
            raise ValueError(f"No ssa values in BasicBlock {self.number}")
    
    def __str__(self) -> str:
        return f"BasicBlock {self.number}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def get_instructions(self) -> list[str]:
        return self.join_instructions + self.instructions

class IntermediateRepresentation:
    def __init__(self):
        const_basic_block = BasicBlock(0)
        basic_block_1 = BasicBlock(1)
        basic_block_1.parent_blocks.add(const_basic_block)
        self.basic_blocks = {0: const_basic_block, 1: basic_block_1}
        self._next_free_basic_block_number = 2
        self._ssa_value = 1
        self._ssa_const_value = -1
        self.ssa_associations = dict()

        self._predefined_functions = {"InputNum", "OutputNum", "OutputNewLine"}
        self.functions = self._predefined_functions.copy()

        self.join_basic_block = None

    def create_new_ssa_value(self) -> int:
        self._ssa_value += 1
        return self._ssa_value - 1
    
    def create_new_ssa_const_value(self) -> int:
        self._ssa_const_value -= 1
        return self._ssa_const_value + 1
    
    def create_new_basic_block(self) -> BasicBlock:
        if self._next_free_basic_block_number == 5:
            print("debug")
        new_basic_block = BasicBlock(self._next_free_basic_block_number)
        self.basic_blocks[self._next_free_basic_block_number] = new_basic_block
        self._next_free_basic_block_number += 1
        return new_basic_block
    
    def save_join_basic_block(self, basic_block: BasicBlock) -> None:
        self.join_basic_block = basic_block
    
    def get_potential_join_basic_block(self, basic_block: BasicBlock) -> BasicBlock:
        jbb = self.join_basic_block
        self.join_basic_block = None
        return jbb if jbb else basic_block
    
    def find_ssa_value_of(self, operand: int | str) -> int:
        if operand in self.ssa_associations:
            return self.ssa_associations[operand]
        else: # should technically only be constants
            ssa_value = self.create_new_ssa_const_value()
            self.ssa_associations[operand] = ssa_value
            instruction = Instruction(ssa_value, Operator.CONST, operand)
            self.basic_blocks[0].instructions.append(instruction)
            #emit(instruction)
            return ssa_value
    
    def emit_program(self) -> None:
        for basic_block in self.basic_blocks.keys():
            emit(f"# Basic Block {basic_block}")
            for instruction in self.basic_blocks[basic_block].get_instructions():
                emit(instruction)
            emit()
        
        
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
    def expression(basic_block: BasicBlock) -> int:
        ssa_value = term(basic_block)
        while file_stream.peek_token() in ["+", "-"]:
            token = file_stream.peek_token()
            file_stream.next_token()
            new_ssa_value = intrepr.create_new_ssa_value()
            match token:
                case Token.PLUS:
                    add_instruction = Instruction(new_ssa_value, Operator.ADD, ssa_value, term(basic_block))
                    basic_block.instructions.append(add_instruction)
                    #emit(add_instruction)
                case Token.MINUS:
                    sub_instruction = Instruction(new_ssa_value, Operator.SUB, ssa_value, term(basic_block))
                    basic_block.instructions.append(sub_instruction)
                    #emit(sub_instruction)
            ssa_value = new_ssa_value
        return ssa_value

    def term(basic_block: BasicBlock) -> int:
        ssa_value = factor(basic_block)
        while file_stream.peek_token() in ["*", "/"]:
            token = file_stream.peek_token()
            file_stream.next_token()
            new_ssa_value = intrepr.create_new_ssa_value()
            match token:
                case Token.MULTIPLY:
                    mul_instruction = Instruction(new_ssa_value, Operator.MUL, ssa_value, factor(basic_block))
                    basic_block.instructions.append(mul_instruction)
                case Token.DIVIDE:
                    div_instruction = Instruction(new_ssa_value, Operator.DIV, ssa_value, factor(basic_block))
                    basic_block.instructions.append(div_instruction)
            ssa_value = new_ssa_value
        return ssa_value
    
    def factor(basic_block: BasicBlock) -> int:
        ssa_value = None
        if file_stream.peek_token() == "(":
            file_stream.next_token()
            ssa_value = expression(basic_block)
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
        return intrepr.find_ssa_value_of(operand)
    
    ## STATEMENT PARSING
    def assignment(basic_block: BasicBlock) -> int:
        file_stream.next_token() # consume Keyword.LET
        variable_name = file_stream.peek_token()
        file_stream.next_token() # consume variable_name
        if file_stream.peek_token() != Token.ASSIGNMENT:
            raise SyntaxError(f"Expected \'<-\' following \'let\' + identifier \'{variable_name}\'")
        file_stream.next_token() # consume Token.ASSIGNMENT
        intrepr.ssa_associations[variable_name] = expression(basic_block) # map variable_name to SSA value
        basic_block.variables_to_most_recent_ssa_value[variable_name] = intrepr.ssa_associations[variable_name]
        return intrepr.ssa_associations[variable_name]
    
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

    def if_then_else_fi(if_basic_block: BasicBlock) -> None:
        then_basic_block = intrepr.create_new_basic_block()
        join_basic_block = intrepr.create_new_basic_block()
        intrepr.basic_blocks[then_basic_block.number] = then_basic_block
        intrepr.basic_blocks[join_basic_block.number] = join_basic_block
        then_basic_block.parent_blocks.add(if_basic_block)
        join_basic_block.parent_blocks.add(then_basic_block)

        # processing condition clause
        if file_stream.peek_token() != Keyword.IF:
            raise SyntaxError(f"Expected \'if\' token")
        file_stream.next_token() # consume Keyword.IF
        left_ssa_value = expression(if_basic_block)
        compare_operator_token = file_stream.peek_token()
        file_stream.next_token()
        right_ssa_value = expression(if_basic_block)
        compare_ssa_value = intrepr.create_new_ssa_value()
        compare_instruction = Instruction(compare_ssa_value, Operator.CMP, left_ssa_value, right_ssa_value)
        intrepr.basic_blocks[if_basic_block.number] = if_basic_block
        if_basic_block.instructions.append(compare_instruction)
        branch_operator = Operator.map_comparison_operator(compare_operator_token)
        branch_instruction_ssa_value = intrepr.create_new_ssa_value()

        # processing then clause
        if file_stream.peek_token() != Keyword.THEN:
            raise SyntaxError(f"Expected \'then\' token")
        file_stream.next_token() # consume Keyword.THEN
        statement_sequence(then_basic_block)

        # processing else clause
        else_basic_block = None
        if file_stream.peek_token() == Keyword.ELSE:
            file_stream.next_token() # consume Keyword.ELSE
            else_basic_block = intrepr.create_new_basic_block()
            else_basic_block.parent_blocks.add(if_basic_block)
            intrepr.basic_blocks[else_basic_block.number] = else_basic_block
            statement_sequence(else_basic_block)
            branch_from_if_to_else_instruction = Instruction(branch_instruction_ssa_value, branch_operator, compare_ssa_value, else_basic_block.get_starting_ssa_value())
            if_basic_block.instructions.append(branch_from_if_to_else_instruction)
            #emit(branch_to_else_clause_instruction)
        if file_stream.peek_token() != Keyword.FI:
            raise SyntaxError(f"Expected \'fi\' token")
        file_stream.next_token()

        # post-processing join block, branching then block to join block
        intrepr.save_join_basic_block(join_basic_block)
        join_basic_block.parent_blocks.add(then_basic_block)
        join_basic_block.parent_blocks.add(else_basic_block) if else_basic_block else None
        
        for variable_name in then_basic_block.variables_to_most_recent_ssa_value.keys():
            then_ssa_value = then_basic_block.variables_to_most_recent_ssa_value[variable_name]
            branched_ssa_value = None
            if else_basic_block and variable_name in else_basic_block.variables_to_most_recent_ssa_value:
                branched_ssa_value = else_basic_block.variables_to_most_recent_ssa_value[variable_name]
            else:
                branched_ssa_value = if_basic_block.variables_to_most_recent_ssa_value[variable_name]
            phi_instruction = Instruction(intrepr.create_new_ssa_value(), Operator.PHI, then_ssa_value, branched_ssa_value)
            join_basic_block.join_instructions.append(phi_instruction)
            intrepr.ssa_associations[variable_name] = phi_instruction.ssa_value
            join_basic_block.variables_to_most_recent_ssa_value[variable_name] = phi_instruction.ssa_value

        if else_basic_block:
            branch_instruction_from_then_to_join = Instruction(intrepr.create_new_ssa_value(), Operator.BRANCH, join_basic_block.get_starting_ssa_value())
            then_basic_block.instructions.append(branch_instruction_from_then_to_join)
        else:
            branch_instruction_from_if_to_join = Instruction(intrepr.create_new_ssa_value(), Operator.BRANCH, join_basic_block.get_starting_ssa_value())
            if_basic_block.instructions.append(branch_instruction_from_if_to_join)

        return
        
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

    def statement_sequence(basic_block: BasicBlock) -> None:
        statement(basic_block)
        basic_block = intrepr.get_potential_join_basic_block(basic_block)
        while file_stream.peek_token() == Token.SEPARATOR:
            file_stream.next_token()
            if file_stream.peek_token() == Token.CLOSE_BRACE:
                break
            statement(basic_block) # discard subsequent ssa values; want STARTING only
        return

    def statement(basic_block: BasicBlock) -> None:
        while True: # do-while loop for parsing statements
            basic_block = intrepr.get_potential_join_basic_block(basic_block)
            match file_stream.peek_token():
                case Keyword.LET:
                        assignment(basic_block)
                case Keyword.IF:
                        if_then_else_fi(basic_block)
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
                case Keyword.CALL:
                    function_call()
                case Keyword.VOID | Keyword.FUNCTION:
                    function_declaration()
                case _:
                    break
        return
        
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
            statement_sequence(intrepr.basic_blocks[1])
            if file_stream.peek_token() != Token.CLOSE_BRACE:
                raise SyntaxError(f"Expected \'}}\' to close statement sequence")
            file_stream.next_token()
        if file_stream.peek_token() != ".":
            raise SyntaxError(f"Expected \'.\' to end computation")
    
    # Begin compilation
    file_stream.next_token()
    computation()
    intrepr.emit_program()
    #emit()

    # closing all output streams except STDOUT
    close_files(*output_streams[1:])
    

if __name__ == "__main__":
    compile()