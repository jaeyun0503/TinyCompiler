from sys import stdout as STDOUT
from enum import Enum
from pathlib import Path

from tokenizer import Keyword, Token, SourceCode
from output import OutStream, open_files, close_files

# MODIFY TO TARGET SOURCE CODE FILE
SOURCE_CODE_FILE_NAME = "FULL_if_else_nested.txt"

# CONSOLE OUTPUT TOGGLE
OUTPUT_TO_CONSOLE = False


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
    SET_PARAMETER_2 = "setpar2"
    SET_PARAMETER_3 = "setpar3"
    EMPTY_INSTRUCTION = "\\<empty\\>"

    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def get_parameter(cls, parameter_number: int):
        match parameter_number:
            case 1:
                return Operator.GET_PARAMETER_1
            case 2:
                return Operator.GET_PARAMETER_2
            case 3:
                return Operator.GET_PARAMETER_3
            case _:
                raise ValueError(f"Invalid parameter number {parameter_number}")
            
    @classmethod
    def set_parameter(cls, argument_number: int):
        match argument_number:
            case 1:
                return Operator.SET_PARAMETER_1
            case 2:
                return Operator.SET_PARAMETER_2
            case 3:
                return Operator.SET_PARAMETER_3
            case _:
                raise ValueError(f"Invalid argument number {argument_number}")

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

class Instruction:
    def __init__(self, ssa_value: int = None, operator: Operator = None,
                 operand_1: str = "", operand_2: str = "",
                 operand_1_variable_name: str = "", operand_2_variable_name: str = ""):
        self.ssa_value = ssa_value  
        self.operator = operator
        self.operand_1 = operand_1
        self.operand_2 = operand_2
        self.operand_1_variable_name = operand_1_variable_name
        self.operand_2_variable_name = operand_2_variable_name

    def associate_variable_names(self, variable_name_to_ssa_value: dict[str, int]) -> None:
        return None
        for variable_name, ssa_value in variable_name_to_ssa_value.items():
            if self.operand_1 == ssa_value:
                self.operand_1_variable_name = variable_name
            if self.operand_2 == ssa_value:
                self.operand_2_variable_name = variable_name

    def __str__(self) -> str:
        if self.operator == Operator.CONST:
            return f"{self.ssa_value}: {self.operator}{self.operand_1}"
        else:
            return f"{self.ssa_value}: {self.operator} {self.operand_1} {self.operand_2}".rstrip()
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Instruction):
            return NotImplemented
        return self.operator == other.operator and self.operand_1 == other.operand_1 and self.operand_2 == other.operand_2

class BasicBlock:
    def __init__(self, block_number: int = None):
        self.number: int = block_number
        self.instructions: list[Instruction] = []
        self.join_instructions: list[Instruction] = []
        self.dominated_by: list[BasicBlock] = []
        self.variable_names_to_latest_ssa_values: dict[str, int] = dict()
        self.is_if_merge_point: bool = False
        self.contains_if: bool = False
        self.contains_while: bool = False
    
    def get_starting_ssa_value(self) -> int:
        if self.join_instructions:
            return self.join_instructions[0].ssa_value
        elif self.instructions:
            return self.instructions[0].ssa_value
        else:
            return None
    
    def __str__(self) -> str:
        return f"BasicBlock {self.number}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def get_instructions(self) -> list[Instruction]:
        return self.join_instructions + self.instructions
    
class Function:
    def __init__(self):
        self.name: str = None
        self.formal_parameters: list = []
        self.variables: list = []
        self.has_return_value: bool = True

        self.const_basic_block: BasicBlock = None
        self.first_basic_block: BasicBlock = None
        self.basic_blocks: dict[int, BasicBlock] = []

        self.program: Program = None
        self.constants_to_ssa_values: dict[str, int] = dict()
        self.join_basic_block: BasicBlock = None
        self.control_flow_stack: list[BasicBlock] = []

    def save_join_basic_block(self, bb: BasicBlock) -> None:
        self.join_basic_block = bb
    
    def determine_join_basic_block(self, bb: BasicBlock) -> BasicBlock:
        jbb = self.join_basic_block
        self.join_basic_block = None
        return jbb if jbb else bb
    
    def save_join_basic_block(self, bb: BasicBlock) -> None:
        self.join_basic_block = bb
    
    def determine_join_basic_block(self, bb: BasicBlock) -> BasicBlock:
        jbb = self.join_basic_block
        self.join_basic_block = None
        return jbb if jbb else bb
    
    def find_var_ssa_value(self, var: str) -> int:
        """
        To be removed
        """
        if self.constants_to_ssa_values[var] is int.imag:
            self.program.warn(f"var {var} was uninitialized before use in function {self.name}")
            LITERAL_ZERO = "0"
            self.find_const_ssa_value(LITERAL_ZERO)
        else:
            return self.constants_to_ssa_values[var]
    
    def find_const_ssa_value(self, const: str) -> int:
        if const in self.constants_to_ssa_values:
            return self.constants_to_ssa_values[const]
        else: # should technically only be constants
            ssa_value = self.program.generate_ssa_const_value()
            self.constants_to_ssa_values[const] = ssa_value
            instruction = Instruction(ssa_value, Operator.CONST, const)
            self.const_basic_block.instructions.append(instruction)
            return ssa_value
    
    def find_subexpression_ssa_value(self, basic_block: BasicBlock, instruction: Instruction) -> int:
        for basic_block in [basic_block] + basic_block.dominated_by[::-1]:
            for i in basic_block.get_instructions()[::-1]:
                if i == instruction:
                    return i.ssa_value
        return None
    
    def fill_empty_instructions(self) -> None:
        for basic_block in self.basic_blocks.values():
            if not basic_block.get_instructions():
                basic_block.instructions.append(Instruction(self.program.generate_ssa_value(), Operator.EMPTY_INSTRUCTION))
    
    def __repr__(self) -> str:
        return f"{self.name}({tuple(self.formal_parameters)})"

dot_outstream = None
class Dot:
    """
    Generates digraph representation of IR using DOT graph language
    """
    SAPPHIRE = "#2f47a1"
    PRAIRIE_SAND = "#9b3b24"
    DARK_SIENNA = "#3f0096"
    FOREST_GREEN = "#1c782f"
    LIGHT_SAPPHIRE = "#6b80d4"
    LIGHT_PRAIRIE_SAND = "#d26f4f"
    LIGHT_SIENNA_VIOLET = "#8b5fe0"
    LIGHT_FOREST_GREEN = "#4ebf63"

    PENWIDTH = 5

    def __init__(self):
        self.functions: list[Function] = []
        self.basic_blocks: set[BasicBlock] = set()
        self.arrows: list = []
    
    def declare_function(self, f: Function) -> None:
        self.functions.append(f)

    def declare_block(self, bb: BasicBlock) -> None:
        self.basic_blocks.add(bb)

    def declare_arrow(self, from_bb: BasicBlock, to_bb: BasicBlock) -> None:
        self.arrows.append(self.translate_arrow(from_bb, to_bb))
    
    def declare_fall_through_arrow(self, from_bb: BasicBlock, to_bb: BasicBlock) -> None:
        self.arrows.append(self.translate_fall_through_arrow(from_bb, to_bb))

    def declare_branch_arrow(self, from_bb: BasicBlock, to_bb: BasicBlock) -> None:
        self.arrows.append(self.translate_branch_arrow(from_bb, to_bb))
    
    def declare_dom_arrow(self, from_bb: BasicBlock, to_bb: BasicBlock) -> None:
        self.arrows.append(self.translate_dom_arrow(from_bb, to_bb))

    def emit_program(self) -> None:
        print(self.__str__(), file=dot_outstream)

    def __str__(self) -> str:
        content = []
        for f in self.functions:
            content.append(Dot.translate_function(f))
        for bb in self.basic_blocks:
            if bb.is_if_merge_point:
                content.append(Dot.translate_join_block(bb))
            else:
                content.append(Dot.translate_block(bb))
        for f in self.functions:
            content.append(Dot.translate_function_relation_arrow(f, f.const_basic_block))
        content.extend(self.arrows)
        for bb in self.basic_blocks:
            for curser_bb in bb.dominated_by:
                content.append(Dot.translate_dom_arrow(curser_bb, bb))
        newline = '\n'
        return f"digraph G {{\n{newline.join(content)}\n}}"
        # return (f"digraph G {{\n"
        #         f"  bgcolor=\"black\";\n"
        #         f"  node [shape=record, style=filled, fillcolor=\"#202020\", fontcolor=\"white\", color=\"white\"];\n"
        #         f"  edge [color=\"white\", fontcolor=\"white\"];\n"
        #         f"{newline.join(content)}\n"
        #         f"}}"
        # )

    @classmethod
    def translate_function(cls, f: Function) -> str:
        return f"function{f.name} [shape=trapezium, label=\"F[{f.name}]\"];"
    
    @classmethod
    def translate_function_relation_arrow(cls, f: Function, bb: BasicBlock) -> str:
        return f"function{f.name}:s -> bb{bb.number}:n [arrowhead=icurve, penwidth=2];"

    @classmethod
    def translate_block(cls, bb: BasicBlock) -> str:
        bbn = bb.number
        instructions = "|".join([str(i) for i in bb.get_instructions()])
        return f"bb{bbn} [shape=record,label=\"<b>BB[{bbn}]|{{{instructions}}}\"];"
    
    @classmethod
    def translate_join_block(cls, bb: BasicBlock) -> str:
        bbn = bb.number
        instructions = "|".join([str(i) for i in bb.get_instructions()])
        return f"bb{bbn} [shape=record,label=\"<b>join\\nBB[{bbn}]|{{{instructions}}}\"];"    
    
    @classmethod
    def translate_arrow(cls, from_bb: BasicBlock, to_bb: BasicBlock) -> str:
        return f"bb{from_bb.number}:s -> bb{to_bb.number}:n;"
    
    @classmethod
    def translate_fall_through_arrow(cls, from_bb: BasicBlock, to_bb: BasicBlock) -> str:
        return f"bb{from_bb.number}:s -> bb{to_bb.number}:n [label=\"fall-through\", color=\"{Dot.LIGHT_FOREST_GREEN}\", fontcolor=\"{Dot.LIGHT_FOREST_GREEN}\"];"
    
    @classmethod
    def translate_branch_arrow(cls, from_bb: BasicBlock, to_bb: BasicBlock) -> str:
        return f"bb{from_bb.number}:s -> bb{to_bb.number}:n [label=\"branch\", color=\"{Dot.LIGHT_PRAIRIE_SAND}\", fontcolor=\"{Dot.LIGHT_PRAIRIE_SAND}\"];"
    
    @classmethod
    def translate_dom_arrow(cls, from_bb: BasicBlock, to_bb: BasicBlock) -> str:
        return f"bb{from_bb.number}:s -> bb{to_bb.number}:b [label=\"dom\", style=dotted, color=\"{Dot.LIGHT_SAPPHIRE}\", fontcolor=\"{Dot.LIGHT_SAPPHIRE}\"];"

object_outstream = None
class Program:
    def __init__(self):
        self.MAIN_FUNCTION_CONST_BASIC_BLOCK_NUMBER = None
        self.MAIN_FUNCTION_BASIC_BLOCK_NUMBER = None
        self.computation_basic_blocks = dict()
        self._basic_block_counter = 1
        self._ssa_value_counter = 1
        self._ssa_constant_value_counter = -1

        self.UNINITIALIZED_DEFAULT_VALUE = "0"
        self._MAIN_FUNCTION_PREDEFINITION = "_MAIN"
        self.main_function = Function(); self.main_function.name = self._MAIN_FUNCTION_PREDEFINITION
        self.predefined_functions = {"InputNum" : Operator.READ, "OutputNum" : Operator.WRITE, "OutputNewLine" : Operator.WRITE_NEW_LINE}
        self.functions = {self._MAIN_FUNCTION_PREDEFINITION: self.main_function}
        self.serving_function = None

        self.warnings = []

    def generate_ssa_value(self) -> int:
        self._ssa_value_counter += 1
        return self._ssa_value_counter - 1
    
    def peek_next_ssa_value(self) -> int:
        return self._ssa_value_counter
    
    def generate_ssa_const_value(self) -> int:
        self._ssa_constant_value_counter -= 1
        return self._ssa_constant_value_counter + 1
    
    def create_new_basic_block(self) -> BasicBlock:
        new_bb = BasicBlock(self._basic_block_counter)
        self._basic_block_counter += 1
        return new_bb
    
    def find_starting_ssa_value_of_function(self, function_name: str) -> int:
        function: Function = self.functions[function_name]
        return function.first_basic_block.get_instructions()[0].ssa_value
    
    def serve_function(self, function: Function):
        self.functions[function.name] = function
        self.serving_function = function

    def initialize_function_control_flow(self, f: Function) -> None:
        f.program = self
        f.const_basic_block = self.create_new_basic_block()
        f.first_basic_block = self.create_new_basic_block()
        f.basic_blocks = {f.const_basic_block.number : f.const_basic_block,
                          f.first_basic_block.number : f.first_basic_block}
        
    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def emit_program(self) -> None:
        for warning in self.warnings:
            emit(warning, outstream=object_outstream)
        emit(outstream=object_outstream)
        for function in self.functions.values():
            for basic_block in function.basic_blocks.values():
                emit(f"Basic Block {basic_block.number}", outstream=object_outstream)
                for instruction in basic_block.get_instructions():
                    if instruction.operator is Operator.EMPTY_INSTRUCTION:
                        instruction.operator = "<empty>"
                    emit(instruction, outstream=object_outstream)
                emit(outstream=object_outstream)
        
def emit(content: str = "", outstream=None) -> None:
    print(content, file=outstream)

def compile(file_name: str = None) -> None:
    # Data structures for parsing, IR generation, and visualization
    SOURCE_CODE_FOLDER_NAME = "tiny"
    file_stream = SourceCode(Path(__file__).resolve().parent / SOURCE_CODE_FOLDER_NAME / Path(SOURCE_CODE_FILE_NAME))
    program = Program()
    dot = Dot()

    # Output handling
    OBJECT_FOLDER_NAME = "o"
    DOT_FOLDER_NAME = "dot"
    Path(Path(__file__).resolve().parent / OBJECT_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    Path(Path(__file__).resolve().parent / DOT_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    OBJECT_FILE_PATH = Path(__file__).resolve().parent / OBJECT_FOLDER_NAME / Path(SOURCE_CODE_FILE_NAME).with_suffix(".o")
    DOT_FILE_PATH = Path(__file__).resolve().parent / DOT_FOLDER_NAME / Path(SOURCE_CODE_FILE_NAME).with_suffix(".dot")
    output_file_paths = [OBJECT_FILE_PATH, DOT_FILE_PATH]
    output_streams = open_files(output_file_paths)
    object_output_streams = [STDOUT] if OUTPUT_TO_CONSOLE else []
    dot_output_streams = [STDOUT] if OUTPUT_TO_CONSOLE else []
    object_output_streams.append(output_streams[0])
    dot_output_streams.append(output_streams[1])
    global object_outstream
    global dot_outstream
    object_outstream = OutStream(*object_output_streams)
    dot_outstream = OutStream(*dot_output_streams)

    ## EXPRESSION PARSING
    def expression(basic_block: BasicBlock) -> int:
        ssa_value = term(basic_block)
        while file_stream.peek_token() in ["+", "-"]:
            token = file_stream.peek_token()
            file_stream.next_token()
            match token:
                case Token.PLUS:
                    add_instruction = Instruction(None, Operator.ADD, ssa_value, term(basic_block))
                    # add_instruction.associate_variable_names(basic_block.variable_names_to_latest_ssa_values)
                    ssa_value = program.serving_function.find_subexpression_ssa_value(basic_block, add_instruction)
                    if ssa_value is None: # create new add instruction between two operands
                        ssa_value = program.generate_ssa_value()
                        add_instruction.ssa_value = ssa_value
                        basic_block.instructions.append(add_instruction)
                case Token.MINUS:
                    sub_instruction = Instruction(None, Operator.SUB, ssa_value, term(basic_block))
                    # sub_instruction.associate_variable_names(basic_block.variable_names_to_latest_ssa_values)
                    ssa_value = program.serving_function.find_subexpression_ssa_value(basic_block, sub_instruction)
                    if ssa_value is None: # create new sub instruction between two operands
                        ssa_value = program.generate_ssa_value()
                        sub_instruction.ssa_value = ssa_value
                        basic_block.instructions.append(sub_instruction)
        return ssa_value

    def term(basic_block: BasicBlock) -> int:
        ssa_value = factor(basic_block)
        while file_stream.peek_token() in ["*", "/"]:
            token = file_stream.peek_token()
            file_stream.next_token()
            match token:
                case Token.MULTIPLY:
                    mul_instruction = Instruction(None, Operator.MUL, ssa_value, factor(basic_block))
                    # mul_instruction.associate_variable_names(basic_block.variable_names_to_latest_ssa_values)
                    new_ssa_value = program.serving_function.find_subexpression_ssa_value(basic_block, mul_instruction)
                    if new_ssa_value is None:
                        new_ssa_value = program.generate_ssa_value()
                        mul_instruction.ssa_value = new_ssa_value
                        basic_block.instructions.append(mul_instruction)
                case Token.DIVIDE:
                    div_instruction = Instruction(None, Operator.DIV, ssa_value, factor(basic_block))
                    # div_instruction.associate_variable_names(basic_block.variable_names_to_latest_ssa_values)
                    new_ssa_value = program.serving_function.find_subexpression_ssa_value(basic_block, div_instruction)
                    if new_ssa_value is None:
                        new_ssa_value = program.generate_ssa_value()
                        div_instruction.ssa_value = new_ssa_value
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
            ssa_value = number(basic_block)
        else:
            raise SyntaxError(f"Factor no digits")
        return ssa_value

    def number(basic_block: BasicBlock) -> int:
        operand = file_stream.peek_token()
        ssa_value = None
        if operand == Keyword.CALL:
            return function_call(basic_block)
        elif operand.isdigit():
            ssa_value = program.serving_function.find_const_ssa_value(operand)
        else:
            if operand not in basic_block.variable_names_to_latest_ssa_values:
                program.warn(f"WARNING: var {operand} was uninitialized before usage in function {program.serving_function.name}")
                ssa_value = program.serving_function.find_const_ssa_value(program.UNINITIALIZED_DEFAULT_VALUE)
                basic_block.variable_names_to_latest_ssa_values[operand] = ssa_value
            else:
                ssa_value = basic_block.variable_names_to_latest_ssa_values[operand]

        file_stream.next_token()
        return ssa_value
    
    ## STATEMENT PARSING
    def assignment(basic_block: BasicBlock) -> int:
        file_stream.next_token() # consume Keyword.LET
        variable_name = file_stream.peek_token()
        file_stream.next_token() # consume variable_name
        if file_stream.peek_token() != Token.ASSIGNMENT:
            raise SyntaxError(f"Expected \'<-\' following \'let\' + identifier \'{variable_name}\'")
        file_stream.next_token() # consume Token.ASSIGNMENT
        program.serving_function.constants_to_ssa_values[variable_name] = expression(basic_block) # map variable_name to SSA value
        basic_block.variable_names_to_latest_ssa_values[variable_name] = program.serving_function.constants_to_ssa_values[variable_name] # record in basic block
        return program.serving_function.constants_to_ssa_values[variable_name]
    
    def function_call(basic_block: BasicBlock) -> None:
        file_stream.next_token() # consume Keyword.CALL
        function_name = file_stream.peek_token()
        file_stream.next_token() # consume function_name
        function_arguments = []
        if file_stream.peek_token() == Token.OPEN_PARENTHESES:
            file_stream.next_token() # consume Token.OPEN_PARENTHESES
            function_arguments.append(expression(basic_block))
            while file_stream.peek_token() == Token.COMMA:
                file_stream.next_token()
                function_arguments.append(expression(basic_block))
            if file_stream.peek_token() != Token.CLOSE_PARENTHESES:
                raise SyntaxError(f"Expected \')\' to close called function arguments")
            file_stream.next_token()
        jump_subroutine_instruction = None
        if function_name in program.predefined_functions:
            operation_name = program.predefined_functions[function_name]
            function_call_instruction = Instruction(program.generate_ssa_value(), operation_name, *function_arguments)
            # function_call_instruction.associate_variable_names(basic_block.variable_names_to_latest_ssa_values)
            basic_block.instructions.append(function_call_instruction)
        else:
            if function_name not in program.functions:
                raise NotImplementedError(f"Function {function_name} does not exist")
            jump_subroutine_instruction = Instruction(program.generate_ssa_value(), Operator.JUMP_SUBROUTINE, program.find_starting_ssa_value_of_function(function_name))
            argument_count = 1
            for argument in function_arguments:
                set_parameter_instruction = Instruction(program.generate_ssa_value(), Operator.set_parameter(argument_count), argument)
                basic_block.instructions.append(set_parameter_instruction)
                argument_count += 1
            basic_block.instructions.append(jump_subroutine_instruction)
        return jump_subroutine_instruction.ssa_value if jump_subroutine_instruction else None

    def if_then_else_fi(if_basic_block: BasicBlock) -> None:
        if_basic_block.contains_if = True
        then_basic_block: BasicBlock = program.create_new_basic_block()
        join_basic_block: BasicBlock = program.create_new_basic_block()
        join_basic_block.is_if_merge_point = True
        function = program.serving_function
        function.basic_blocks[then_basic_block.number] = then_basic_block
        function.basic_blocks[join_basic_block.number] = join_basic_block
        then_basic_block.variable_names_to_latest_ssa_values = if_basic_block.variable_names_to_latest_ssa_values.copy()
        join_basic_block.variable_names_to_latest_ssa_values = if_basic_block.variable_names_to_latest_ssa_values.copy()
        then_basic_block.dominated_by = [if_basic_block] + if_basic_block.dominated_by
        join_basic_block.dominated_by = [if_basic_block] + if_basic_block.dominated_by
        dot.declare_block(if_basic_block)
        dot.declare_block(then_basic_block)
        dot.declare_block(join_basic_block)
        dot.declare_fall_through_arrow(if_basic_block, then_basic_block)

        # processing condition clause
        if file_stream.peek_token() != Keyword.IF:
            raise SyntaxError(f"Expected \'if\' token")
        file_stream.next_token() # consume Keyword.IF
        left_ssa_value = expression(if_basic_block)
        compare_operator_token = file_stream.peek_token()
        file_stream.next_token()
        right_ssa_value = expression(if_basic_block)
        compare_ssa_value = program.generate_ssa_value()
        compare_instruction = Instruction(compare_ssa_value, Operator.CMP, left_ssa_value, right_ssa_value)
        # compare_instruction.associate_variable_names(if_basic_block.variable_names_to_latest_ssa_values)
        function.basic_blocks[if_basic_block.number] = if_basic_block
        if_basic_block.instructions.append(compare_instruction)
        branch_operator = Operator.map_comparison_operator(compare_operator_token)
        branch_instruction_ssa_value = program.generate_ssa_value()

        # processing then clause
        if file_stream.peek_token() != Keyword.THEN:
            raise SyntaxError(f"Expected \'then\' token")
        file_stream.next_token() # consume Keyword.THEN
        statement_sequence(then_basic_block)
        function.control_flow_stack.append(then_basic_block)
        if then_basic_block.contains_if:
            function.control_flow_stack.pop()

        # processing else clause
        else_basic_block = None
        if file_stream.peek_token() == Keyword.ELSE:
            file_stream.next_token() # consume Keyword.ELSE
            else_basic_block = program.create_new_basic_block()
            function.basic_blocks[else_basic_block.number] = else_basic_block
            else_basic_block.variable_names_to_latest_ssa_values = if_basic_block.variable_names_to_latest_ssa_values.copy()
            else_basic_block.dominated_by = [if_basic_block] + if_basic_block.dominated_by
            statement_sequence(else_basic_block)
            dot.declare_block(else_basic_block)
            if not else_basic_block.get_instructions():
                else_basic_block.instructions.append(Instruction(program.generate_ssa_value(), Operator.EMPTY_INSTRUCTION))
            branch_from_if_to_else_instruction = Instruction(branch_instruction_ssa_value, branch_operator, compare_ssa_value, else_basic_block.get_starting_ssa_value())
            if_basic_block.instructions.append(branch_from_if_to_else_instruction)
            function.control_flow_stack.append(else_basic_block)
            if else_basic_block.contains_if:
                function.control_flow_stack.pop()    
        else:
            function.control_flow_stack.append(if_basic_block)

        if file_stream.peek_token() != Keyword.FI:
            raise SyntaxError(f"Expected \'fi\' token")
        file_stream.next_token()

        if else_basic_block:
            dot.declare_branch_arrow(if_basic_block, else_basic_block)
        else:
            dot.declare_branch_arrow(if_basic_block, join_basic_block)
        head_else_basic_block = else_basic_block
        else_basic_block = function.control_flow_stack.pop()
        then_basic_block = function.control_flow_stack.pop()
        for variable_name in program.serving_function.variables + program.serving_function.formal_parameters:
            if variable_name not in then_basic_block.variable_names_to_latest_ssa_values and variable_name not in else_basic_block.variable_names_to_latest_ssa_values:
                continue
            then_ssa_value = None
            branched_ssa_value = None
            if variable_name not in then_basic_block.variable_names_to_latest_ssa_values:
                then_ssa_value = program.serving_function.find_const_ssa_value(program.UNINITIALIZED_DEFAULT_VALUE)
                program.warn(f"WARNING: var {variable_name} uninitialized before use in function {program.serving_function.name}")
            else:
                then_ssa_value = then_basic_block.variable_names_to_latest_ssa_values[variable_name]
            if variable_name not in else_basic_block.variable_names_to_latest_ssa_values:
                branched_ssa_value = program.serving_function.find_const_ssa_value(program.UNINITIALIZED_DEFAULT_VALUE)
                program.warn(f"WARNING: var {variable_name} uninitialized before use in function {program.serving_function.name}")
            else:
                branched_ssa_value = else_basic_block.variable_names_to_latest_ssa_values[variable_name]
            if then_ssa_value == branched_ssa_value:
                continue
            phi_instruction = Instruction(program.generate_ssa_value(), Operator.PHI, then_ssa_value, branched_ssa_value)
            # phi_instruction.associate_variable_names(then_basic_block.variable_names_to_latest_ssa_values)
            join_basic_block.join_instructions.append(phi_instruction)
            function.constants_to_ssa_values[variable_name] = phi_instruction.ssa_value
            join_basic_block.variable_names_to_latest_ssa_values[variable_name] = phi_instruction.ssa_value

        if not then_basic_block.get_instructions():
            then_basic_block.instructions.append(Instruction(program.generate_ssa_value(), Operator.EMPTY_INSTRUCTION))
        if not join_basic_block.get_instructions():
            join_basic_block.instructions.append(Instruction(program.generate_ssa_value(), Operator.EMPTY_INSTRUCTION))
        if head_else_basic_block:
            branch_instruction_from_then_to_join = Instruction(program.generate_ssa_value(), Operator.BRANCH, join_basic_block.get_starting_ssa_value())
            then_basic_block.instructions.append(branch_instruction_from_then_to_join)
            dot.declare_branch_arrow(then_basic_block, join_basic_block)
            dot.declare_fall_through_arrow(else_basic_block, join_basic_block)
        else:
            branch_instruction_from_if_to_join = Instruction(program.generate_ssa_value(), Operator.map_comparison_operator(compare_operator_token), compare_ssa_value, join_basic_block.get_starting_ssa_value())
            if_basic_block.instructions.append(branch_instruction_from_if_to_join)
            dot.declare_fall_through_arrow(then_basic_block, join_basic_block)
        function.control_flow_stack.append(join_basic_block)
        function.save_join_basic_block(join_basic_block)
        return
        
    def _rewrite_do_ssa_values(variable_name: str, new_ssa_value: int, while_basic_block: BasicBlock) -> None:
        basic_blocks = []
        for basic_block in program.basic_blocks.values():
            if while_basic_block in basic_block.dominated_by:
                basic_blocks.append(basic_block)

        for basic_block in basic_blocks:
            for instruction in basic_block.get_instructions():
                if instruction.operand_1_variable_name == variable_name and instruction.operand_1 == while_basic_block.variable_names_to_latest_ssa_values[variable_name]:
                    instruction.operand_1 = new_ssa_value
                if instruction.operand_2_variable_name == variable_name and instruction.operand_2 == while_basic_block.variable_names_to_latest_ssa_values[variable_name]:
                    instruction.operand_2 = new_ssa_value
        return None


    def while_do_od(while_basic_block: BasicBlock) -> None:
        while_basic_block.contains_while = True
        do_basic_block = program.create_new_basic_block()
        od_basic_block = program.create_new_basic_block()
        program.basic_blocks[do_basic_block.number] = do_basic_block
        program.basic_blocks[od_basic_block.number] = od_basic_block
        do_basic_block.variable_names_to_latest_ssa_values = while_basic_block.variable_names_to_latest_ssa_values.copy()
        od_basic_block.variable_names_to_latest_ssa_values = while_basic_block.variable_names_to_latest_ssa_values.copy()
        do_basic_block.dominated_by = [while_basic_block] + while_basic_block.dominated_by
        od_basic_block.dominated_by = [while_basic_block] + while_basic_block.dominated_by
        dot.declare_block(while_basic_block)
        dot.declare_block(do_basic_block)
        dot.declare_block(od_basic_block)
        dot.declare_fall_through_arrow(while_basic_block, do_basic_block)

        if file_stream.peek_token() != Keyword.WHILE:
            raise SyntaxError(f"Expected \'while\' token")
        file_stream.next_token() # consume Keyword.WHILE
        left_ssa_value = expression(while_basic_block)
        compare_operator_token = file_stream.peek_token()
        file_stream.next_token()
        right_ssa_value = expression(while_basic_block)
        compare_instruction = Instruction(program.generate_ssa_value(), Operator.CMP, left_ssa_value, right_ssa_value)
        # compare_instruction.associate_variable_names(while_basic_block.variable_names_to_latest_ssa_values)
        while_basic_block.instructions.append(compare_instruction)
        branch_operator = Operator.map_comparison_operator(compare_operator_token)
        branch_instruction_from_while_to_od = Instruction(program.generate_ssa_value(), branch_operator, compare_instruction.ssa_value, None)
        while_basic_block.instructions.append(branch_instruction_from_while_to_od)
        
        if file_stream.peek_token() != Keyword.DO:
            raise SyntaxError(f"Expected \'do\' token")
        file_stream.next_token() # consume Keyword.DO
        statement_sequence(do_basic_block)
        if not do_basic_block.contains_while and not do_basic_block.contains_if:
            program.control_flow_stack.append(do_basic_block)
        do_basic_block = program.control_flow_stack.pop()
        for variable_name in do_basic_block.variable_names_to_latest_ssa_values:
            if do_basic_block.variable_names_to_latest_ssa_values[variable_name] == while_basic_block.variable_names_to_latest_ssa_values[variable_name]:
                continue
            while_ssa_value = while_basic_block.variable_names_to_latest_ssa_values[variable_name]
            do_ssa_value = do_basic_block.variable_names_to_latest_ssa_values[variable_name]
            phi_instruction = Instruction(program.generate_ssa_value(), Operator.PHI, do_ssa_value, while_ssa_value)
            # phi_instruction.associate_variable_names(while_basic_block.variable_names_to_latest_ssa_values)
            while_basic_block.join_instructions.append(phi_instruction)
            _rewrite_do_ssa_values(variable_name, phi_instruction.ssa_value, while_basic_block)
        
        do_basic_block.instructions.append(Instruction(program.generate_ssa_value(), Operator.BRANCH, while_basic_block.get_starting_ssa_value()))
        branch_instruction_from_while_to_od.operand_2 = program.peek_next_ssa_value()
        dot.declare_branch_arrow(do_basic_block, while_basic_block)
        dot.declare_branch_arrow(while_basic_block, od_basic_block)
        program.control_flow_stack.append(od_basic_block)
        program.save_join_basic_block(od_basic_block)
        
        if file_stream.peek_token() != Keyword.OD:
            raise SyntaxError(f"Expected \'od\' token")
        file_stream.next_token()
        return
    
    def returning(basic_block: BasicBlock) -> None:
        if file_stream.peek_token() != Keyword.RETURN:
            raise SyntaxError(f"Expected \'return\' token")
        file_stream.next_token() # consume Keyword.RETURN
        result_ssa_value = ""
        if file_stream.peek_token() != Token.SEPARATOR:
            result_ssa_value = expression(basic_block)
        return_instruction = Instruction(program.generate_ssa_value(), Operator.RETURN, result_ssa_value)
        basic_block.instructions.append(return_instruction)
        return

    def statement_sequence(basic_block: BasicBlock) -> None:
        statement(basic_block)
        basic_block = program.serving_function.determine_join_basic_block(basic_block)
        while file_stream.peek_token() == Token.SEPARATOR:
            file_stream.next_token()
            if file_stream.peek_token() == Token.CLOSE_BRACE:
                break
            statement(basic_block)
        return

    def statement(basic_block: BasicBlock) -> None:
        while True: # loop for parsing statements
            basic_block = program.serving_function.determine_join_basic_block(basic_block)
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
                    while_do_od(basic_block)
                case Keyword.OD:
                    break
                case Keyword.RETURN:
                    returning(basic_block)
                case Keyword.CALL:
                    function_call(basic_block)
                case Keyword.VOID | Keyword.FUNCTION:
                    function_declaration()
                case Token.SEPARATOR:
                    file_stream.next_token()
                    continue
                case Token.CLOSE_BRACE:
                    break
                case _:
                    print("Fall-through case in statement parsing")
                    break
        return
        
    def variable_declaration() -> list[str]:
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

    def function_body(function: Function) -> None:
        variable_names = variable_declaration()
        if file_stream.peek_token() != Token.OPEN_BRACE:
            raise SyntaxError(f"Expected \'{{\' to start statement sequence")
        file_stream.next_token()
        program.serving_function.variables = variable_names
        statement_sequence(function.first_basic_block)
        function.fill_empty_instructions()
        if file_stream.peek_token() != Token.CLOSE_BRACE:
            raise SyntaxError(f"Expected \'}}\' to end statement sequence")
        file_stream.next_token()
        return variable_names

    def formal_parameters(function: Function) -> None:
        if file_stream.peek_token() != Token.OPEN_PARENTHESES:
            raise SyntaxError(f"Expected \'(\' to start formal params")
        file_stream.next_token()
        parameters = []
        parameter_count = 0
        if file_stream.peek_token() != Token.CLOSE_PARENTHESES:
            while True:
                parameters.append(file_stream.peek_token())
                file_stream.next_token()
                parameter_count += 1
                getpar_instruction = Instruction(function.program.generate_ssa_value(), Operator.get_parameter(parameter_count))
                function.first_basic_block.instructions.append(getpar_instruction)
                function.first_basic_block.variable_names_to_latest_ssa_values[parameters[-1]] = getpar_instruction.ssa_value
                if file_stream.peek_token() != Token.COMMA:
                    break
                file_stream.next_token()
        if file_stream.peek_token() != Token.CLOSE_PARENTHESES:
            raise SyntaxError(f"Expected \')\' to close formal params")
        file_stream.next_token()
        function.formal_parameters = parameters
        return None

    def function_declaration(function: Function) -> None:
        if file_stream.peek_token() == Keyword.VOID:
            function.has_return_value = False
            file_stream.next_token()
        file_stream.next_token() # consume Keyword.FUNCTION
        function.name = file_stream.peek_token()
        program.initialize_function_control_flow(function)
        program.serve_function(function)
        dot.declare_block(function.const_basic_block)
        dot.declare_block(function.first_basic_block)
        dot.declare_arrow(function.const_basic_block, function.first_basic_block)
        file_stream.next_token() # consume function_name
        formal_parameters(function)
        file_stream.next_token() # consume ;
        function_body(function)
        if file_stream.peek_token() != Token.SEPARATOR:
            raise SyntaxError(f"Expected \';\' to end function body")
        file_stream.next_token()
        return function
        
    def computation() -> None:
        """
        Parses entire source code
        """
        if file_stream.peek_token() == Keyword.MAIN:
            file_stream.next_token()
            main_variables = variable_declaration()
            while file_stream.peek_token() in [Keyword.VOID, Keyword.FUNCTION]:
                function = Function()
                dot.declare_function(function)
                function_declaration(function)
            if file_stream.peek_token() != Token.OPEN_BRACE:
                raise SyntaxError(f"Expected \'{{\' to open statement sequence")
            file_stream.next_token()
            dot.declare_function(program.main_function)
            program.initialize_function_control_flow(program.main_function)
            program.serve_function(program.main_function)
            program.main_function.variables = main_variables
            dot.declare_block(program.main_function.const_basic_block)
            dot.declare_block(program.main_function.first_basic_block)
            dot.declare_arrow(program.main_function.const_basic_block, program.main_function.first_basic_block)
            statement_sequence(program.main_function.first_basic_block)
            program.main_function.fill_empty_instructions()
            if file_stream.peek_token() != Token.CLOSE_BRACE:
                raise SyntaxError(f"Expected \'}}\' to close statement sequence")
            file_stream.next_token()
        if file_stream.peek_token() != ".":
            raise SyntaxError(f"Expected \'.\' to end computation")
    
    # Begin compilation
    file_stream.next_token()
    computation()

    # Handling output
    dot.emit_program()
    program.emit_program()

    # closing all output streams except STDOUT
    close_files(*output_streams)
    

if __name__ == "__main__":
    compile()