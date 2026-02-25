from sys import stdout as STDOUT
from enum import Enum
from pathlib import Path

from tokenizer import Keyword, Token, SourceCode
from output import OutStream, open_files, close_files

# MODIFY TO TARGET SOURCE CODE FILE
SOURCE_CODE_FILE_NAME = "while_example.txt"

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
            
    @classmethod
    def map_math_operator(cls, token: str):
        match token:
            case Token.PLUS:
                return Operator.ADD
            case Token.MINUS:
                return Operator.SUB
            case Token.MULTIPLY:
                return Operator.MUL
            case Token.DIVIDE:
                return Operator.DIV
            case _:
                raise SyntaxError(f"Expected valid math operator token")

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
        self.commons = [self]

    def is_common_subexpression(self, other) -> bool:
        if not isinstance(other, Instruction):
            return NotImplemented
        return self.operator == other.operator and self.operand_1 == other.operand_1 and self.operand_2 == other.operand_2

    def has_same_variables(self, other) -> bool:
        if not isinstance(other, Instruction):
            return NotImplemented
        return self.operand_1_variable_name == other.operand_1_variable_name and self.operand_2_variable_name == other.operand_2_variable_name

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
        self.variable_names_ssa_values: dict[str, int] = dict()
        self.vars_exps: dict[str, str] = dict()
        self.is_join_block: bool = False
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
                if i.is_common_subexpression(instruction):
                    i.commons.append(instruction)
                    instruction.operand_2_variable_name = self.program.variables_used.pop()
                    instruction.operand_1_variable_name = self.program.variables_used.pop()
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
            if bb.is_join_block:
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

class UninitializedWarning():
    def __init__(self, variable_name: str, function_name: str):
        self.variable_name = variable_name
        self.function_name = function_name
    
    def __str__(self):
        return f"WARNING: var {self.variable_name} is uninitialized before use in function {self.function_name}"

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
        self.focused_function = None

        self.variables_used = []
        self.term_count = 1
        self.expression_has_terms = False
        self.variable_name_ready = False

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
    
    def focus_function(self, function: Function):
        self.functions[function.name] = function
        self.focused_function = function

    def initialize_function_control_flow(self, f: Function) -> None:
        f.program = self
        f.const_basic_block = self.create_new_basic_block()
        f.first_basic_block = self.create_new_basic_block()
        f.basic_blocks = {f.const_basic_block.number : f.const_basic_block,
                          f.first_basic_block.number : f.first_basic_block}
        
    # def save_variable(self, variable_name: str) -> None:
    #     self.variable_used = variable_name
    
    def load_variable(self) -> str:
        self.variable_name_ready = False
        return self.variables_used.pop()

    def warn(self, message: Warning) -> None:
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

    def consume_token(*tokens) -> None:
        if file_stream.peek_token() not in tokens:
            raise SyntaxError(f"Expected token(s): {{{', '.join(tokens)}}}")
        file_stream.next_token()
        return

    def _create_math_instruction(basic_block: BasicBlock, ssa_value: int, operator_token: Token) -> int:
        math_instruction = Instruction(None, Operator.map_math_operator(operator_token), ssa_value, term(basic_block))
        ssa_value = program.focused_function.find_subexpression_ssa_value(basic_block, math_instruction)
        if ssa_value is None:
            ssa_value = program.generate_ssa_value()
            math_instruction.ssa_value = ssa_value
            math_instruction.operand_2_variable_name = program.variables_used.pop()
            math_instruction.operand_1_variable_name = program.variables_used.pop()
            basic_block.instructions.append(math_instruction)
        return ssa_value

    ## EXPRESSION PARSING
    def expression(basic_block: BasicBlock) -> int:
        ssa_value = expression_lower(basic_block)
        if program.term_count == 1:
            program.variable_name_ready = True
        # program.term_count = 1
        return ssa_value

    def expression_lower(basic_block: BasicBlock) -> int:
        ssa_value = term(basic_block)
        operator_token = ""
        while file_stream.peek_token() in [Token.PLUS, Token.MINUS]:
            if program.term_count > 1:
                program.variables_used.append("")
            program.term_count += 1
            operator_token = file_stream.peek_token()
            file_stream.next_token()
            ssa_value = _create_math_instruction(basic_block, ssa_value, operator_token)
        return ssa_value

    def term(basic_block: BasicBlock) -> int:
        ssa_value = factor(basic_block)
        operator_token = ""
        while file_stream.peek_token() in ["*", "/"]:
            if program.term_count > 1:
                program.variables_used.append("")
            program.term_count += 1
            operator_token = file_stream.peek_token()
            file_stream.next_token()
            ssa_value = _create_math_instruction(basic_block, ssa_value, operator_token)
        return ssa_value
    
    def factor(basic_block: BasicBlock) -> int:
        ssa_value = None
        if file_stream.peek_token() == "(":
            file_stream.next_token()
            ssa_value = expression_lower(basic_block)
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
            program.variables_used.append("") # //FLAG,  might be an issue here when an expression includes a func call
            return function_call(basic_block)
        elif operand.isdigit():
            program.variables_used.append(str(operand))
            ssa_value = program.focused_function.find_const_ssa_value(operand)
        else:
            if operand not in basic_block.variable_names_ssa_values:
                program.warn(UninitializedWarning(operand, program.focused_function))
                ssa_value = program.focused_function.find_const_ssa_value(program.UNINITIALIZED_DEFAULT_VALUE)
                basic_block.variable_names_ssa_values[operand] = ssa_value
            else:
                ssa_value = basic_block.variable_names_ssa_values[operand]
            program.variables_used.append(operand)

        file_stream.next_token()
        return ssa_value
    
    ## STATEMENT PARSING
    def _generate_expression_hash(basic_block: BasicBlock, term_count: int) -> str:
        """
        Assumes last term_count-1 instructions were math instructions
        """
        instruction_count = term_count - 1
        buffer = []
        for instruction in basic_block.get_instructions()[::-1][:instruction_count][::-1]:
            buffer.append(str(instruction.commons[-1].operator))
            buffer.append(f":")
            buffer.append(str(instruction.commons[-1].operand_1_variable_name))
            buffer.append(f":")
            buffer.append(str(instruction.commons[-1].operand_2_variable_name))
        return "".join(buffer)

    def assignment(basic_block: BasicBlock) -> int:
        file_stream.next_token() # consume Keyword.LET
        variable_name = file_stream.peek_token()
        file_stream.next_token() # consume variable_name
        if file_stream.peek_token() != Token.ASSIGNMENT:
            raise SyntaxError(f"Expected \'<-\' following \'let\' + identifier \'{variable_name}\'")
        file_stream.next_token() # consume Token.ASSIGNMENT
        program.focused_function.constants_to_ssa_values[variable_name] = expression(basic_block) # map variable_name to SSA value
        expression_hash = None
        if program.variable_name_ready:
            expression_hash = program.load_variable()
            if expression_hash.isalpha() and expression_hash in basic_block.variable_names_ssa_values:
                expression_hash = basic_block.vars_exps[expression_hash]
            program.variable_name_ready = False
        else:
            expression_hash = _generate_expression_hash(basic_block, program.term_count)
            program.term_count = 1
        basic_block.variable_names_ssa_values[variable_name] = program.focused_function.constants_to_ssa_values[variable_name] # record in basic block
        basic_block.vars_exps[variable_name] = expression_hash
        return program.focused_function.constants_to_ssa_values[variable_name]
    
    def function_call(basic_block: BasicBlock) -> None:
        file_stream.next_token() # consume Keyword.CALL
        function_name = file_stream.peek_token()
        file_stream.next_token() # consume function_name
        function_arguments = []
        if file_stream.peek_token() == Token.OPEN_PARENTHESES:
            file_stream.next_token() # consume Token.OPEN_PARENTHESES
            if file_stream.peek_token() != Token.CLOSE_PARENTHESES:
                function_arguments.append(expression(basic_block))
                if program.variable_name_ready:
                    program.load_variable()
                while file_stream.peek_token() == Token.COMMA:
                    file_stream.next_token()
                    function_arguments.append(expression(basic_block))
                    if program.variable_name_ready:
                        program.load_variable()
                if file_stream.peek_token() != Token.CLOSE_PARENTHESES:
                    raise SyntaxError(f"Expected \')\' to close called function arguments")
            file_stream.next_token()
        function_call_ssa_value = None
        if function_name in program.predefined_functions:
            operation_name = program.predefined_functions[function_name]
            function_call_instruction = Instruction(program.generate_ssa_value(), operation_name, *function_arguments)
            basic_block.instructions.append(function_call_instruction)
            function_call_ssa_value = function_call_instruction.ssa_value
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
            function_call_ssa_value = jump_subroutine_instruction.ssa_value
        return function_call_ssa_value

    def _process_condition_clause(basic_block: BasicBlock) -> Instruction:
        if file_stream.peek_token() not in [Keyword.IF, Keyword.WHILE]:
            raise SyntaxError(f"Expected condition token (i.e. \'if\', \'while\'")
        file_stream.next_token() # consume Keyword.WHILE or Keyword.IF
        left_ssa_value = expression(basic_block)
        left_literal = program.load_variable() if program.variable_name_ready else ""
        compare_operator_token = file_stream.peek_token()
        file_stream.next_token()
        right_ssa_value = expression(basic_block)
        right_literal = program.load_variable() if program.variable_name_ready else ""
        compare_instruction = Instruction(program.generate_ssa_value(), Operator.CMP, left_ssa_value, right_ssa_value, left_literal, right_literal)
        basic_block.instructions.append(compare_instruction)
        branch_operator = Operator.map_comparison_operator(compare_operator_token)
        branch_instruction = Instruction(program.generate_ssa_value(), branch_operator, compare_instruction.ssa_value, None)
        basic_block.instructions.append(branch_instruction)
        return branch_instruction
    
    def _push_non_branching_basic_block(basic_block: BasicBlock) -> None:
        if (basic_block.contains_if or basic_block.contains_while) is False:
            program.focused_function.control_flow_stack.append(basic_block)
        return
    
    def _process_then_excerpt(then_basic_block: BasicBlock) -> None:
        if file_stream.peek_token() != Keyword.THEN:
            raise SyntaxError(f"Expected \'then\' token")
        file_stream.next_token() # consume Keyword.THEN
        statement_sequence(then_basic_block)
        _push_non_branching_basic_block(then_basic_block)
        return
    
    def _process_else_excerpt(if_basic_block: BasicBlock) -> None:
        else_basic_block = None
        if file_stream.peek_token() == Keyword.ELSE:
            file_stream.next_token() # consume Keyword.ELSE
            else_basic_block = program.create_new_basic_block()
            program.focused_function.basic_blocks[else_basic_block.number] = else_basic_block
            else_basic_block.variable_names_ssa_values = if_basic_block.variable_names_ssa_values.copy()
            else_basic_block.dominated_by = [if_basic_block] + if_basic_block.dominated_by
            statement_sequence(else_basic_block)
            dot.declare_block(else_basic_block)
            if not else_basic_block.get_instructions():
                else_basic_block.instructions.append(Instruction(program.generate_ssa_value(), Operator.EMPTY_INSTRUCTION))
            _push_non_branching_basic_block(else_basic_block)
        else:
            program.focused_function.control_flow_stack.append(if_basic_block)
        return else_basic_block

    def _get_variable_ssa_value(variable_name: str, basic_block: BasicBlock) -> int:
        ssa_value = None
        if variable_name in basic_block.variable_names_ssa_values:
            ssa_value = basic_block.variable_names_ssa_values[variable_name]
        else:
            ssa_value = program.focused_function.find_const_ssa_value(program.UNINITIALIZED_DEFAULT_VALUE)
            program.warn(UninitializedWarning(variable_name, program.focused_function))
        return ssa_value

    def _process_if_join_block(then_basic_block: BasicBlock,
                            not_then_basic_block: BasicBlock,
                            join_basic_block: BasicBlock) -> None:
        for variable_name in program.focused_function.variables + program.focused_function.formal_parameters:
            if variable_name not in then_basic_block.variable_names_ssa_values and variable_name not in not_then_basic_block.variable_names_ssa_values:
                continue
            then_ssa_value = _get_variable_ssa_value(variable_name, then_basic_block)
            not_then_ssa_value = _get_variable_ssa_value(variable_name, not_then_basic_block)
            if then_ssa_value == not_then_ssa_value:
                continue
            phi_instruction = Instruction(program.generate_ssa_value(), Operator.PHI, then_ssa_value, not_then_ssa_value)
            join_basic_block.join_instructions.append(phi_instruction)
            join_basic_block.variable_names_ssa_values[variable_name] = phi_instruction.ssa_value
        return

    def if_then_else_fi(if_basic_block: BasicBlock) -> None:
        if_basic_block.contains_if = True
        
        then_basic_block_head = None
        then_basic_block_tail = None
        function = program.focused_function

        then_basic_block_head: BasicBlock = program.create_new_basic_block()
        join_basic_block: BasicBlock = program.create_new_basic_block()
        join_basic_block.is_join_block = True

        function.basic_blocks[if_basic_block.number] = if_basic_block
        function.basic_blocks[then_basic_block_head.number] = then_basic_block_head
        function.basic_blocks[join_basic_block.number] = join_basic_block
        then_basic_block_head.variable_names_ssa_values = if_basic_block.variable_names_ssa_values.copy()
        join_basic_block.variable_names_ssa_values = if_basic_block.variable_names_ssa_values.copy()
        then_basic_block_head.vars_exps = if_basic_block.vars_exps.copy()
        join_basic_block.vars_exps = if_basic_block.vars_exps.copy()
        then_basic_block_head.dominated_by = [if_basic_block] + if_basic_block.dominated_by
        join_basic_block.dominated_by = [if_basic_block] + if_basic_block.dominated_by

        # processing condition statement
        compare_branch_instruction = _process_condition_clause(if_basic_block)

        # processing then clause
        _process_then_excerpt(then_basic_block_head)

        # processing else clause
        else_basic_block: BasicBlock = _process_else_excerpt(if_basic_block)

        if file_stream.peek_token() != Keyword.FI:
            raise SyntaxError(f"Expected \'fi\' token")
        file_stream.next_token()

        if 19 in function.control_flow_stack and 16 in function.control_flow_stack:
            print("debug")

        not_then_basic_block_head = else_basic_block
        not_then_basic_block_tail = function.control_flow_stack.pop()
        then_basic_block_tail = function.control_flow_stack.pop()
        _process_if_join_block(then_basic_block_tail, not_then_basic_block_tail, join_basic_block)

        # handle branching
        if not join_basic_block.get_instructions():
            join_basic_block.instructions.append(Instruction(program.generate_ssa_value(), Operator.EMPTY_INSTRUCTION))
        if else_basic_block:
            if not else_basic_block.get_instructions():
                then_basic_block_head.instructions.append(Instruction(program.generate_ssa_value(), Operator.EMPTY_INSTRUCTION))
            compare_branch_instruction.operand_2 = not_then_basic_block_head.get_starting_ssa_value()
            then_basic_block_tail.instructions.append(Instruction(program.generate_ssa_value(), Operator.BRANCH, join_basic_block.get_starting_ssa_value()))
        else:
            compare_branch_instruction.operand_2 = join_basic_block.get_starting_ssa_value()

        function.control_flow_stack.append(join_basic_block)
        function.save_join_basic_block(join_basic_block)

        # dot everything
        dot.declare_block(if_basic_block)
        dot.declare_block(then_basic_block_head)
        dot.declare_block(join_basic_block)
        dot.declare_fall_through_arrow(if_basic_block, then_basic_block_head)
        if else_basic_block:
            dot.declare_block(not_then_basic_block_head)
            dot.declare_branch_arrow(if_basic_block, not_then_basic_block_head)
            dot.declare_branch_arrow(then_basic_block_tail, join_basic_block)
            dot.declare_fall_through_arrow(not_then_basic_block_tail, join_basic_block)
        else:
            dot.declare_branch_arrow(if_basic_block, join_basic_block)
            dot.declare_fall_through_arrow(then_basic_block_tail, join_basic_block)

        return
    
        
    def _rewrite_do_ssa_values(variable_name: str, target_ssa_value: int, while_basic_block: BasicBlock, do_basic_block: BasicBlock, remaining_vars: set[str]) -> None:
        # what's the point of old_ssa_value?
        instruction: Instruction
        new_ssa_value = while_basic_block.variable_names_ssa_values[variable_name]

        basic_blocks: list[BasicBlock] = []
        for basic_block in program.focused_function.basic_blocks.values():
            if basic_block is not while_basic_block and while_basic_block in basic_block.dominated_by:
                basic_blocks.append(basic_block)

        for basic_block in basic_blocks:
            instructions: list[Instruction] = []
            for instruction in basic_block.get_instructions():
                if instruction.operator is Operator.CMP:
                    if instruction.operand_1_variable_name == variable_name:
                        instruction.operand_1 = new_ssa_value
                    if instruction.operand_2_variable_name == variable_name:
                        instruction.operand_2 = new_ssa_value
                elif instruction.operator in [Operator.ADD, Operator.SUB, Operator.MUL, Operator.DIV]:
                    if target_ssa_value not in [instruction.operand_1, instruction.operand_2]:
                        instructions.append(instruction)
                        continue
                    similars: list[Instruction] = []
                    for similar in instruction.commons:
                        if similar.operand_1_variable_name == variable_name:
                            similar.operand_1 = new_ssa_value
                        if similar.operand_2_variable_name == variable_name:
                            similar.operand_2 = new_ssa_value
                        if similar.operand_1_variable_name == variable_name or similar.operand_2_variable_name == variable_name:
                            # common instruction is substituted into instruction's place
                                # take ssa value as well
                            similar.ssa_value = instruction.ssa_value
                            instructions.append(similar)
                            # reconstruct instruction's common list without common
                            if len(instruction.commons) > 1:
                                new_commons = [c for c in instruction.commons if c != similar] # // FLAG, != or is not?
                                instruction.commons = [instruction]
                                instruction = new_commons[0]
                                instruction.commons = new_commons
                                prev_instr_ssa = similar.ssa_value
                                new_instr_ssa = program.generate_ssa_value()
                                instruction.ssa_value = new_instr_ssa
                                for vn in remaining_vars:
                                    do_basic_block.variable_names_ssa_values[vn] = new_instr_ssa
                                instructions.append(instruction)
                            break
                        similars.append(similar)
                    if similars == instruction.commons:
                        instructions.append(instruction)
                else:
                    instructions.append(instruction)
            basic_block.instructions = instructions
        return None


    def while_do_od(while_basic_block: BasicBlock) -> None:
        pre_while_basic_block: BasicBlock = None
        do_basic_block_head: BasicBlock = program.create_new_basic_block()
        do_basic_block_tail: BasicBlock = None
        od_basic_block: BasicBlock = program.create_new_basic_block()

        if while_basic_block.get_instructions():
            pre_while_basic_block = while_basic_block
            pre_while_basic_block.contains_while = True
            while_basic_block = program.create_new_basic_block()
            while_basic_block.variable_names_ssa_values = pre_while_basic_block.variable_names_ssa_values.copy()
            while_basic_block.vars_exps = pre_while_basic_block.vars_exps.copy()
            while_basic_block.dominated_by = [pre_while_basic_block] + pre_while_basic_block.dominated_by
        
        while_basic_block.contains_while = True
        while_basic_block.is_join_block = True
        function: Function = program.focused_function
        function.basic_blocks[do_basic_block_head.number] = do_basic_block_head
        function.basic_blocks[od_basic_block.number] = od_basic_block
        do_basic_block_head.variable_names_ssa_values = while_basic_block.variable_names_ssa_values.copy()
        od_basic_block.variable_names_ssa_values = while_basic_block.variable_names_ssa_values.copy()
        do_basic_block_head.vars_exps = while_basic_block.vars_exps.copy()
        do_basic_block_head.dominated_by = [while_basic_block] + while_basic_block.dominated_by
        od_basic_block.dominated_by = [while_basic_block] + while_basic_block.dominated_by

        branch_instruction_from_while_to_od = _process_condition_clause(while_basic_block)
        
        if file_stream.peek_token() != Keyword.DO:
            raise SyntaxError(f"Expected \'do\' token")
        file_stream.next_token() # consume Keyword.DO
        statement_sequence(do_basic_block_head)
        _push_non_branching_basic_block(do_basic_block_head)

        do_basic_block_tail = function.control_flow_stack.pop()

        while_basic_block.vars_exps = do_basic_block_tail.vars_exps.copy()
        wbbvs = while_basic_block.variable_names_ssa_values.copy()
        variable_queue = function.variables + function.formal_parameters
        while variable_queue:
            variable_name = variable_queue[0]
            variable_queue = variable_queue[1:]
            if variable_name not in while_basic_block.variable_names_ssa_values and variable_name not in do_basic_block_tail.variable_names_ssa_values:
                continue
            while_ssa_value = _get_variable_ssa_value(variable_name, while_basic_block)
            do_ssa_value = _get_variable_ssa_value(variable_name, do_basic_block_tail)
            if while_ssa_value == do_ssa_value:
                continue
            phi_instruction = Instruction(program.generate_ssa_value(), Operator.PHI, do_ssa_value, while_ssa_value)
            while_basic_block.join_instructions.append(phi_instruction)

            old_ssa_value = wbbvs[variable_name]
            shared_ssa_value = do_basic_block_tail.variable_names_ssa_values[variable_name]
            while_basic_block.variable_names_ssa_values[variable_name] = phi_instruction.ssa_value
            target_hash = while_basic_block.vars_exps[variable_name]
            vars_sharing_ssa = set()
            for var, ssa in do_basic_block_tail.variable_names_ssa_values.items():
                if ssa == shared_ssa_value:
                    vars_sharing_ssa.add(var)
            vars_sharing_hash = set()
            for var, exp in do_basic_block_tail.vars_exps.items():
                if exp == target_hash:
                    vars_sharing_hash.add(var)
            for var in vars_sharing_hash:
                while_basic_block.variable_names_ssa_values[var] = phi_instruction.ssa_value
            remaining_vars = vars_sharing_ssa - vars_sharing_hash
            variable_queue = [var for var in variable_queue if var not in vars_sharing_hash]
            _rewrite_do_ssa_values(variable_name, old_ssa_value, while_basic_block, do_basic_block_tail, remaining_vars)
        
        do_basic_block_tail.instructions.append(Instruction(program.generate_ssa_value(), Operator.BRANCH, while_basic_block.get_starting_ssa_value()))
        if not od_basic_block.get_instructions():
            od_basic_block.instructions.append(Instruction(program.generate_ssa_value(), Operator.EMPTY_INSTRUCTION))
        branch_instruction_from_while_to_od.operand_2 = od_basic_block.get_starting_ssa_value()
        od_basic_block.variable_names_ssa_values = while_basic_block.variable_names_ssa_values.copy()
        od_basic_block.vars_exps = while_basic_block.vars_exps.copy()

        function.control_flow_stack.append(od_basic_block)
        function.save_join_basic_block(od_basic_block)

        # dot everything
        if pre_while_basic_block:
            dot.declare_arrow(pre_while_basic_block, while_basic_block)
        dot.declare_block(while_basic_block)
        dot.declare_block(do_basic_block_head)
        dot.declare_block(od_basic_block)
        dot.declare_fall_through_arrow(while_basic_block, do_basic_block_head)
        dot.declare_branch_arrow(do_basic_block_tail, while_basic_block)
        dot.declare_branch_arrow(while_basic_block, od_basic_block)
        
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
            if program.variable_name_ready:
                program.load_variable()
        return_instruction = Instruction(program.generate_ssa_value(), Operator.RETURN, result_ssa_value)
        basic_block.instructions.append(return_instruction)
        return

    def statement_sequence(basic_block: BasicBlock) -> None:
        statement(basic_block)
        basic_block = program.focused_function.determine_join_basic_block(basic_block)
        while file_stream.peek_token() == Token.SEPARATOR:
            file_stream.next_token()
            if file_stream.peek_token() == Token.CLOSE_BRACE:
                break
            statement(basic_block)
        return

    def statement(basic_block: BasicBlock) -> None:
        while True: # loop for parsing statements
            basic_block = program.focused_function.determine_join_basic_block(basic_block)
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
        program.focused_function.variables = variable_names
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
                function.first_basic_block.variable_names_ssa_values[parameters[-1]] = getpar_instruction.ssa_value
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
        program.focus_function(function)
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
            program.focus_function(program.main_function)
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