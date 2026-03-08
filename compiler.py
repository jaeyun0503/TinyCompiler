from sys import stdout as STDOUT
from pathlib import Path
from enum import Enum
from queue import Queue

from tokenizer import Keyword, Token, SourceCode
from output import OutStream, open_files, close_files

# MODIFY TO TARGET SOURCE CODE FILE
SOURCE_CODE_FILE_NAME = "if_else_simple.txt"

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
    MOV = "mov"

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

class Color:
    r_colors = {1: "#FFB3BA", 2: "#FFDFBA", 3: "#FFFFBA", 4: "#BAFFC9", 5: "#BAE1FF"}
    v_color = "#A1A1A1"

    @classmethod
    def get_color(cls, register_number: int) -> str:
        return cls.r_colors.get(register_number, cls.v_color)

class RegisterAllocator:
    RELEVANT_OPERATORS = {
            Operator.CMP, Operator.PHI, Operator.JUMP_SUBROUTINE,
            Operator.ADD, Operator.SUB, Operator.MUL, Operator.DIV,
            Operator.READ, Operator.EMPTY_INSTRUCTION, Operator.MOV
    }
    BRANCH_OPERATORS = {
            Operator.BRANCH, Operator.BRANCH_NOT_EQUAL, Operator.BRANCH_EQUAL,
            Operator.BRANCH_LESS_THAN, Operator.BRANCH_LESS_THAN_EQUAL,
            Operator.BRANCH_GREATER_THAN, Operator.BRANCH_GREATER_THAN_EQUAL
    }
    MAX_REGISTERS = 5

    @classmethod
    def regno_to_regname(cls, register_number: int) -> str:
        """
        Given a register number, returns string literal of register name
        R represents register
        V represents virtual register (memory)
        Register numbers are not reset when starting V
        register_number: int
        @return: str
        """
        if type(register_number) is str:
            return register_number
        if register_number < 0:
            return register_number
        if register_number <= cls.MAX_REGISTERS:
            return f"R{register_number}"
        else:
            return f"V{register_number}"

    @classmethod
    def welsh_powell_color(cls, al: dict[int, set[int]]) -> dict[int, int]:
        """
        Takes adjacency list and returns graph coloring
        Graph coloring is mapping of int node to int color
        al: dict[int, set[int]]
        @return: dict[int, int]
        """
        NUMBER_OF_NODES = len(al.keys())
        degrees_nodes: dict[int, set[int]] = dict()
        for node, neighbors in al.items():
            degree = len(neighbors)
            if degree not in degrees_nodes:
                degrees_nodes[degree] = set()
            degrees_nodes[degree].add(node)
        
        sorted_nodes: list[int] = []
        for node in sorted(degrees_nodes.keys(), reverse=True):
            sorted_nodes.extend(sorted(degrees_nodes[node])) # extra sorting for reproducibility

        nodes_colors: dict[int, int] = dict()
        colored_nodes = set()
        color = 1
        while len(colored_nodes) < NUMBER_OF_NODES:
            for node in sorted_nodes:
                if node in colored_nodes:
                    continue
                if all(nodes_colors.get(neighbor, None) != color for neighbor in al[node]):
                    nodes_colors[node] = color
                    colored_nodes.add(node)
            color += 1
        
        return nodes_colors
    

ig_outstream = None
class InterferenceGraph:
    def __init__(self):
        self.adjacency_list: dict[int, set[int]] = dict()

    def declare_edge(self, node_1: int, node_2: int) -> None:
        if node_1 not in self.adjacency_list:
            self.adjacency_list[node_1] = set()
        if node_2 not in self.adjacency_list:
            self.adjacency_list[node_2] = set()
        self.adjacency_list[node_1].add(node_2)
        self.adjacency_list[node_2].add(node_1)
        return
    
    def emit_graph(self, nodes_registers: dict[int, int]) -> None:
        dot_string = InterferenceGraph.translate_adjacency_list_to_dot(self.adjacency_list, nodes_registers)
        emit(dot_string, outstream=ig_outstream)
        return None
    
    @classmethod
    def translate_adjacency_list_to_dot(cls, interference: dict[int, set[int]], nodes_registers: dict[int, int]) -> str:
        """
        Generate a DOT-format string for an interference graph
        described by an adjacency list: dict[int, set[int]].

        Nodes are integers, edges are undirected.
        """
        lines = ["graph InterferenceGraph {", "\tlayout=circo;", "\tnode [shape=circle, style=filled];"]
        seen = set()

        for node, neighbors in interference.items():
            lines.append(f"\t{node};")  # ensure node appears even if isolated
            for neighbor in neighbors:
                if node == neighbor:
                    continue  # avoid self-loops
                edge = tuple(sorted((node, neighbor))) # node to node
                if edge not in seen:
                    seen.add(edge)
                    # node_1_register = nodes_registers[edge[0]]
                    # node_2_register = nodes_registers[edge[1]]
                    # node_1_name = RegisterAllocator.regno_to_regname(node_1_register)
                    # node_2_name = RegisterAllocator.regno_to_regname(node_2_register)
                    # lines.append(f"\t{node_1_name} -- {node_2_name};")
                    lines.append(f"\t{edge[0]} -- {edge[1]};")

        lines.append("\n")
        for node in interference.keys():
            register = nodes_registers[node]
            lines.append(f"\t{node} [fillcolor=\"{Color.get_color(register)}\"];")
        lines.append("}")
        return "\n".join(lines)


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
        """
        Checks if instruction has the same operator and operands
        """
        if not isinstance(other, Instruction):
            return NotImplemented
        return self.operator == other.operator and self.operand_1 == other.operand_1 and self.operand_2 == other.operand_2

    def has_same_variables(self, other) -> bool:
        """
        Checks if instruction uses the same literals for operands
        """
        if not isinstance(other, Instruction):
            return NotImplemented
        return self.operand_1_variable_name == other.operand_1_variable_name and self.operand_2_variable_name == other.operand_2_variable_name

    def __str__(self) -> str:
        if self.ssa_value == "∞":
            return f"{self.operator} {self.operand_1} {self.operand_2}".rstrip()
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
        self.join_instructions: list[Instruction] = []
        self.instructions: list[Instruction] = []
        self.branch_instructions: list[Instruction] = []
        self.dominated_by: list[BasicBlock] = []
        self.variable_names_ssa_values: dict[str, int] = dict()
        self.vars_exps: dict[str, str] = dict()
        self.is_if_block: bool = False
        self.is_pre_while_block: bool = False
        self.is_while_block: bool = False
        self.is_fi_block: bool = False
        self.is_od_block: bool = False
        self.if_block: BasicBlock = None
        self.while_block: BasicBlock = None
        self.od_block: BasicBlock = None
        self.do_tail_block: BasicBlock = None
        self.parents: set[BasicBlock] = set()
        self.children: set[BasicBlock] = set()
    
    def get_starting_ssa_value(self) -> int:
        if self.join_instructions:
            return self.join_instructions[0].ssa_value
        elif self.instructions:
            return self.instructions[0].ssa_value
        elif self.branch_instructions:
            return self.branch_instructions[0].ssa_value
        else:
            return None
    
    def __str__(self) -> str:
        return f"BasicBlock {self.number}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def get_instructions(self) -> list[Instruction]:
        return self.join_instructions + self.instructions + self.branch_instructions
    
class Function:
    def __init__(self):
        self.name: str = None
        self.formal_parameters: list = []
        self.variables: list = []
        self.has_return_value: bool = True

        self.const_basic_block: BasicBlock = None
        self.first_basic_block: BasicBlock = None
        self.basic_blocks: dict[int, BasicBlock] = []
        self.last_basic_block: BasicBlock = None

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
                    instruction.operand_2_variable_name = self.program.literals_used.pop()
                    instruction.operand_1_variable_name = self.program.literals_used.pop()
                    return i.ssa_value
        return None
    
    def fill_empty_instructions(self) -> None:
        for basic_block in self.basic_blocks.values():
            if not basic_block.get_instructions():
                basic_block.instructions.append(Instruction(self.program.generate_ssa_value(), Operator.EMPTY_INSTRUCTION))
    
    def __repr__(self) -> str:
        return f"{self.name}({tuple(self.formal_parameters)})"

dot_outstream = None
regdot_outstream = None
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
        self.arrows: list[str] = []
    
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

    def emit_regalloc_program(self) -> None:
        print(self.__str__(), file=regdot_outstream)

    def __str__(self) -> str:
        content = []
        for f in self.functions:
            content.append(Dot.translate_function(f))
        for bb in self.basic_blocks:
            if bb.is_fi_block or (bb.is_while_block and not bb.is_pre_while_block):
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
    
class ControlFlow:
    def __init__(self):
        self.paths: list[tuple[BasicBlock, BasicBlock]] = []
        self.paths_reversed: list[tuple[BasicBlock, BasicBlock]] = []
        self.src_dests_norm: dict[BasicBlock, set[BasicBlock]] = dict()
        self.src_dests_reversed: dict[BasicBlock, set[BasicBlock]] = dict()

    def add_path(self, from_bb: BasicBlock, to_bb: BasicBlock) -> None:
        self.paths.append((from_bb, to_bb))
        self.paths_reversed.append((to_bb, from_bb))
        if from_bb not in self.src_dests_norm:
            self.src_dests_norm[from_bb] = set()
        self.src_dests_norm[from_bb].add(to_bb)
        if to_bb not in self.src_dests_reversed:
            self.src_dests_reversed[to_bb] = set()
        self.src_dests_reversed[to_bb].add(from_bb)

    @classmethod
    def reverse_paths(cls, paths: list[tuple[BasicBlock, BasicBlock]]) -> list[tuple[BasicBlock, BasicBlock]]:
        return [(to_bb, from_bb) for from_bb, to_bb in paths]

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
        self.input_num_hash_counter = 1

        self.literals_used = []
        self.term_count = 1
        self.expression_has_terms = False
        self.literal_ready = False

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
        new_bb.instructions.append(Instruction(self.generate_ssa_value(), Operator.EMPTY_INSTRUCTION))
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
        f.last_basic_block = f.first_basic_block
        f.basic_blocks = {f.const_basic_block.number : f.const_basic_block,
                          f.first_basic_block.number : f.first_basic_block}
        
    # def save_variable(self, variable_name: str) -> None:
    #     self.variable_used = variable_name
    
    def load_literal(self) -> str:
        self.literal_ready = False
        return self.literals_used.pop()

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
                    if instruction.operator == "<empty>":
                        instruction.operator = Operator.EMPTY_INSTRUCTION
                emit(outstream=object_outstream)
        
def emit(content: str = "", outstream=None) -> None:
    print(content, file=outstream)

def compile(file_name: str = None) -> None:
    # Data structures for parsing, IR generation, and visualization
    SOURCE_CODE_FOLDER_NAME = "tiny"
    file_stream = SourceCode(Path(__file__).resolve().parent / SOURCE_CODE_FOLDER_NAME / Path(SOURCE_CODE_FILE_NAME))
    program = Program()
    dot = Dot()
    ig = InterferenceGraph()
    cf = ControlFlow()

    # Output handling
    OBJECT_FOLDER_NAME = "o"
    DOT_FOLDER_NAME = "dot"
    IG_FOLDER_NAME = "ig"
    REGDOT_FOLDER_NAME = "regdot"
    Path(Path(__file__).resolve().parent / OBJECT_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    Path(Path(__file__).resolve().parent / DOT_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    Path(Path(__file__).resolve().parent / IG_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    Path(Path(__file__).resolve().parent / REGDOT_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    OBJECT_FILE_PATH = Path(__file__).resolve().parent / OBJECT_FOLDER_NAME / Path(SOURCE_CODE_FILE_NAME).with_suffix(".o")
    DOT_FILE_PATH = Path(__file__).resolve().parent / DOT_FOLDER_NAME / Path(SOURCE_CODE_FILE_NAME).with_suffix(".dot")
    IG_FILE_PATH = Path(__file__).resolve().parent / IG_FOLDER_NAME / Path(SOURCE_CODE_FILE_NAME).with_suffix(".ig")
    REGDOT_FILE_PATH = Path(__file__).resolve().parent / REGDOT_FOLDER_NAME / Path(SOURCE_CODE_FILE_NAME).with_suffix(".dot")
    output_file_paths = [OBJECT_FILE_PATH, DOT_FILE_PATH, IG_FILE_PATH, REGDOT_FILE_PATH]
    output_streams = open_files(output_file_paths)
    object_output_streams = [STDOUT] if OUTPUT_TO_CONSOLE else []
    dot_output_streams = [STDOUT] if OUTPUT_TO_CONSOLE else []
    ig_output_streams = [STDOUT] if OUTPUT_TO_CONSOLE else []
    regdot_output_streams = [STDOUT] if OUTPUT_TO_CONSOLE else []
    object_output_streams.append(output_streams[0])
    dot_output_streams.append(output_streams[1])
    ig_output_streams.append(output_streams[2])
    regdot_output_streams.append(output_streams[3])

    global object_outstream
    global dot_outstream
    global ig_outstream
    global regdot_outstream
    object_outstream = OutStream(*object_output_streams)
    dot_outstream = OutStream(*dot_output_streams)
    ig_outstream = OutStream(*ig_output_streams)
    regdot_outstream = OutStream(*regdot_output_streams)

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
            math_instruction.operand_2_variable_name = program.literals_used.pop()
            math_instruction.operand_1_variable_name = program.literals_used.pop()
            basic_block.instructions.append(math_instruction)
        return ssa_value

    ## EXPRESSION PARSING
    def expression(basic_block: BasicBlock) -> int:
        ssa_value = expression_lower(basic_block)
        if program.term_count == 1:
            program.literal_ready = True
        # program.term_count = 1
        return ssa_value

    def expression_lower(basic_block: BasicBlock) -> int:
        ssa_value = term(basic_block)
        operator_token = ""
        while file_stream.peek_token() in [Token.PLUS, Token.MINUS]:
            if program.term_count > 1:
                program.literals_used.append("")
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
                program.literals_used.append("")
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
            #program.literals_used.append("") # //FLAG,  might be an issue here when an expression includes a func call
            return function_call(basic_block)
        elif operand.isdigit():
            program.literals_used.append(str(operand))
            ssa_value = program.focused_function.find_const_ssa_value(operand)
        else:
            if operand not in basic_block.variable_names_ssa_values:
                program.warn(UninitializedWarning(operand, program.focused_function))
                ssa_value = program.focused_function.find_const_ssa_value(program.UNINITIALIZED_DEFAULT_VALUE)
                basic_block.variable_names_ssa_values[operand] = ssa_value
            else:
                ssa_value = basic_block.variable_names_ssa_values[operand]
            program.literals_used.append(operand)

        file_stream.next_token()
        return ssa_value
    
    ## STATEMENT PARSING
    def _generate_expression_hash(basic_block: BasicBlock, term_count: int) -> str:
        """
        Assumes last term_count-1 instructions were math instructions
        """
        instruction_count = term_count - 1
        buffer = []
        index = 1
        while index < term_count:
            instruction = basic_block.instructions[-index]
            if instruction.operator is Operator.JUMP_SUBROUTINE:
                index += 1
                continue
            buffer.append(str(instruction.commons[-1].operator))
            buffer.append(f":")
            buffer.append(str(instruction.commons[-1].operand_1_variable_name))
            buffer.append(f":")
            buffer.append(str(instruction.commons[-1].operand_2_variable_name))
            index += 1

        # for instruction in basic_block.get_instructions()[::-1][:instruction_count][::-1]:
        #     buffer.append(str(instruction.commons[-1].operator))
        #     buffer.append(f":")
        #     buffer.append(str(instruction.commons[-1].operand_1_variable_name))
        #     buffer.append(f":")
        #     buffer.append(str(instruction.commons[-1].operand_2_variable_name))
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
        if program.literal_ready:
            expression_hash = program.load_literal()
            if expression_hash.isalpha() and expression_hash in basic_block.variable_names_ssa_values:
                expression_hash = basic_block.vars_exps[expression_hash]
            program.literal_ready = False
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
        function_arguments_hashes = []
        if file_stream.peek_token() == Token.OPEN_PARENTHESES:
            file_stream.next_token() # consume Token.OPEN_PARENTHESES
            if file_stream.peek_token() != Token.CLOSE_PARENTHESES:
                function_arguments.append(expression(basic_block))
                if program.literal_ready:
                    function_arguments_hashes.append(program.load_literal())
                    program.literal_ready = False
                else:
                    expression_hash = _generate_expression_hash(basic_block, program.term_count)
                    program.term_count = 1
                    function_arguments_hashes.append(expression_hash)
                while file_stream.peek_token() == Token.COMMA:
                    file_stream.next_token()
                    function_arguments.append(expression(basic_block))
                    if program.literal_ready:
                        function_arguments_hashes.append(program.load_literal())
                        program.literal_ready = False
                    else:
                        expression_hash = _generate_expression_hash(basic_block, program.term_count)
                        program.term_count = 1
                        function_arguments_hashes.append(expression_hash)
                if file_stream.peek_token() != Token.CLOSE_PARENTHESES:
                    raise SyntaxError(f"Expected \')\' to close called function arguments")
            file_stream.next_token()
        function_call_ssa_value = None
        input_num_hash = f""
        if function_name in program.predefined_functions:
            operation_name = program.predefined_functions[function_name]
            function_call_instruction = Instruction(program.generate_ssa_value(), operation_name, *function_arguments)
            basic_block.instructions.append(function_call_instruction)
            function_call_ssa_value = function_call_instruction.ssa_value
            if function_name == "InputNum":
                input_num_hash = f"{program.input_num_hash_counter}:"
                program.input_num_hash_counter += 1
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
        program.literals_used.append(f"{input_num_hash}{function_name}:{':'.join(str(arg) for arg in function_arguments_hashes)}")
        return function_call_ssa_value

    def _process_condition_clause(basic_block: BasicBlock) -> Instruction:
        if file_stream.peek_token() not in [Keyword.IF, Keyword.WHILE]:
            raise SyntaxError(f"Expected condition token (i.e. \'if\', \'while\'")
        file_stream.next_token() # consume Keyword.WHILE or Keyword.IF
        left_ssa_value = expression(basic_block)
        left_literal = program.load_literal() if program.literal_ready else ""
        compare_operator_token = file_stream.peek_token()
        file_stream.next_token()
        right_ssa_value = expression(basic_block)
        right_literal = program.load_literal() if program.literal_ready else ""
        compare_instruction = Instruction(program.generate_ssa_value(), Operator.CMP, left_ssa_value, right_ssa_value, left_literal, right_literal)
        basic_block.instructions.append(compare_instruction)
        branch_operator = Operator.map_comparison_operator(compare_operator_token)
        branch_instruction = Instruction(program.generate_ssa_value(), branch_operator, compare_instruction.ssa_value, None)
        basic_block.branch_instructions.append(branch_instruction)
        return branch_instruction
    
    def _push_non_branching_basic_block(basic_block: BasicBlock) -> None:
        if (basic_block.is_if_block or basic_block.is_while_block) is False:
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
            if_basic_block.children.add(else_basic_block)
            else_basic_block.parents.add(if_basic_block)
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
        if_basic_block.is_if_block = True
        
        then_basic_block_head = None
        then_basic_block_tail = None
        function = program.focused_function

        then_basic_block_head: BasicBlock = program.create_new_basic_block()
        join_basic_block: BasicBlock = program.create_new_basic_block()
        join_basic_block.is_fi_block = True
        join_basic_block.if_block = if_basic_block

        function.basic_blocks[if_basic_block.number] = if_basic_block
        function.basic_blocks[then_basic_block_head.number] = then_basic_block_head
        function.basic_blocks[join_basic_block.number] = join_basic_block
        then_basic_block_head.variable_names_ssa_values = if_basic_block.variable_names_ssa_values.copy()
        join_basic_block.variable_names_ssa_values = if_basic_block.variable_names_ssa_values.copy()
        then_basic_block_head.vars_exps = if_basic_block.vars_exps.copy()
        join_basic_block.vars_exps = if_basic_block.vars_exps.copy()
        then_basic_block_head.dominated_by = [if_basic_block] + if_basic_block.dominated_by
        join_basic_block.dominated_by = [if_basic_block] + if_basic_block.dominated_by
        if_basic_block.children.add(then_basic_block_head)
        then_basic_block_head.parents.add(if_basic_block)

        # processing condition statement
        compare_branch_instruction = _process_condition_clause(if_basic_block)

        # processing then clause
        _process_then_excerpt(then_basic_block_head)

        # processing else clause
        else_basic_block: BasicBlock = _process_else_excerpt(if_basic_block)

        if file_stream.peek_token() != Keyword.FI:
            raise SyntaxError(f"Expected \'fi\' token")
        file_stream.next_token()

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
            then_basic_block_tail.branch_instructions.append(Instruction(program.generate_ssa_value(), Operator.BRANCH, join_basic_block.get_starting_ssa_value()))
        else:
            compare_branch_instruction.operand_2 = join_basic_block.get_starting_ssa_value()

        then_basic_block_tail.children.add(join_basic_block)
        join_basic_block.parents.add(then_basic_block_tail)
        not_then_basic_block_tail.children.add(join_basic_block)
        join_basic_block.parents.add(not_then_basic_block_tail)

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
        
        # cf everything
        cf.add_path(if_basic_block, then_basic_block_head)
        if else_basic_block:
            cf.add_path(if_basic_block, not_then_basic_block_head)
            cf.add_path(then_basic_block_tail, join_basic_block)
            cf.add_path(not_then_basic_block_tail, join_basic_block)
        else:
            cf.add_path(if_basic_block, join_basic_block)
            cf.add_path(then_basic_block_tail, join_basic_block)

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
            for instruction in basic_block.instructions:
                if instruction.operator is Operator.CMP:
                    if instruction.operand_1_variable_name == variable_name:
                        instruction.operand_1 = new_ssa_value
                    if instruction.operand_2_variable_name == variable_name:
                        instruction.operand_2 = new_ssa_value
                    instructions.append(instruction)
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
                        if not (instruction.is_common_subexpression(instructions[-1]) and instruction.has_same_variables(instructions[-1])):
                            instructions.append(instruction)
                else:
                    instructions.append(instruction)
            basic_block.instructions = instructions
        return None

    def while_do_od(while_basic_block: BasicBlock) -> None:
        function: Function = program.focused_function
        pre_while_basic_block: BasicBlock = None
        do_basic_block_head: BasicBlock = program.create_new_basic_block()
        do_basic_block_tail: BasicBlock = None
        od_basic_block: BasicBlock = program.create_new_basic_block()

        if while_basic_block.get_instructions():
            pre_while_basic_block = while_basic_block
            pre_while_basic_block.is_while_block = True
            pre_while_basic_block.is_pre_while_block = True
            while_basic_block = program.create_new_basic_block()
            function.basic_blocks[while_basic_block.number] = while_basic_block
            while_basic_block.variable_names_ssa_values = pre_while_basic_block.variable_names_ssa_values.copy()
            while_basic_block.vars_exps = pre_while_basic_block.vars_exps.copy()
            while_basic_block.dominated_by = [pre_while_basic_block] + pre_while_basic_block.dominated_by
            pre_while_basic_block.children.add(while_basic_block)
            while_basic_block.parents.add(pre_while_basic_block)
        
        while_basic_block.is_while_block = True
        od_basic_block.is_od_block = True
        function.basic_blocks[do_basic_block_head.number] = do_basic_block_head
        function.basic_blocks[od_basic_block.number] = od_basic_block
        do_basic_block_head.variable_names_ssa_values = while_basic_block.variable_names_ssa_values.copy()
        od_basic_block.variable_names_ssa_values = while_basic_block.variable_names_ssa_values.copy()
        do_basic_block_head.vars_exps = while_basic_block.vars_exps.copy()
        do_basic_block_head.dominated_by = [while_basic_block] + while_basic_block.dominated_by
        od_basic_block.dominated_by = [while_basic_block] + while_basic_block.dominated_by
        while_basic_block.children.add(do_basic_block_head)
        do_basic_block_head.parents.add(while_basic_block)
        while_basic_block.children.add(od_basic_block)
        od_basic_block.parents.add(while_basic_block)

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
        
        do_basic_block_tail.branch_instructions.append(Instruction(program.generate_ssa_value(), Operator.BRANCH, while_basic_block.get_starting_ssa_value()))
        if not od_basic_block.get_instructions():
            od_basic_block.instructions.append(Instruction(program.generate_ssa_value(), Operator.EMPTY_INSTRUCTION))
        branch_instruction_from_while_to_od.operand_2 = od_basic_block.get_starting_ssa_value()
        od_basic_block.variable_names_ssa_values = while_basic_block.variable_names_ssa_values.copy()
        od_basic_block.vars_exps = while_basic_block.vars_exps.copy()

        do_basic_block_tail.children.add(while_basic_block)
        while_basic_block.parents.add(do_basic_block_tail)

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

        # cf everything
        if pre_while_basic_block:
            cf.add_path(pre_while_basic_block, while_basic_block)
        cf.add_path(while_basic_block, do_basic_block_head)
        cf.add_path(do_basic_block_tail, while_basic_block)
        cf.add_path(while_basic_block, od_basic_block)
        
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
            if program.literal_ready:
                program.load_literal()
            program.term_count = 1
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
            program.focused_function.last_basic_block = basic_block
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
        
    def forward_pass() -> None:
        # SSA generation and setup for backward pass
        computation()
        dot.emit_program()
        program.emit_program()
        return

    def interfere_basic_block(live_set: set[int], basic_block: BasicBlock) -> set[int]:
        for instruction in basic_block.instructions[::-1]:
            if instruction.operator not in RegisterAllocator.RELEVANT_OPERATORS:
                continue
            for ssa_value in live_set:
                if ssa_value != instruction.ssa_value:
                    ig.declare_edge(instruction.ssa_value, ssa_value)
            if instruction.operand_1 and instruction.operand_1 > 0:
                live_set.add(instruction.operand_1)
            if instruction.operand_2 and instruction.operand_2 > 0:
                live_set.add(instruction.operand_2)
            if instruction.ssa_value in live_set:
                live_set.remove(instruction.ssa_value)
        used_live_values = set()
        initialized_live_values = set()
        for instruction in basic_block.join_instructions[::-1]:
            if instruction.operator is not Operator.PHI:
                continue
            if instruction.operand_1 and instruction.operand_1 > 0:
                used_live_values.add(instruction.operand_1)
            if instruction.operand_2 and instruction.operand_2 > 0:
                used_live_values.add(instruction.operand_2)
            if type(instruction.ssa_value) == int:
                initialized_live_values.add(instruction.ssa_value)
        
        # node is owned by
        # owner's nodes
        live_set |= (used_live_values - initialized_live_values)
        return live_set
    
    def determine_od_block(while_children: set[BasicBlock]) -> BasicBlock:
        for child in while_children:
            if child.is_od_block:
                return child
        raise ValueError(f"No od block found in while block children")

    def traverse_while_structure(while_block: BasicBlock) -> set[BasicBlock]:
        members: set[BasicBlock] = set()
        worklist: Queue[BasicBlock] = Queue()
        worklist.put(while_block)
        od_block = determine_od_block(cf.src_dests_norm[while_block])
        while not worklist.empty():
            basic_block = worklist.get()
            dests = cf.src_dests_norm[basic_block]
            for dest in dests:
                if dest in [while_block, od_block] or dest in members:
                    continue
                members.add(dest)
                worklist.put(dest)
        return members

    def remove_subsets(whiles_structs: dict[BasicBlock, set[BasicBlock]]) -> dict[BasicBlock, set[BasicBlock]]:
        while_blocks = list(whiles_structs.keys())
        result = {}
        for while_block in while_blocks:
            struct = whiles_structs[while_block]
            # keep struct only if it is not a strict subset of some other set
            if not any(struct < whiles_structs[other] for other in while_blocks if other != while_block):
                result[while_block] = struct
        return result

    def gather_top_while_structures(while_blocks: list[BasicBlock]) -> dict[BasicBlock, set[BasicBlock]]:
        while_structs = dict()
        for while_block in while_blocks:
            while_structs[while_block] = traverse_while_structure(while_block)
        top_while_structs = remove_subsets(while_structs)
        return top_while_structs

    def gather_live_values(blocks: set[BasicBlock]) -> set[int]:
        live_values = set()
        for block in blocks:
            for instruction in block.get_instructions():
                if instruction.operator not in RegisterAllocator.RELEVANT_OPERATORS:
                    continue
                    # if instruction.operand_1:
                    #     live_values.add(instruction.operand_1)
                    # if instruction.operand_2:
                    #     live_values.add(instruction.operand_2)
                if instruction.operand_1 and instruction.operand_1 > 0:
                    live_values.add(instruction.operand_1)
                if instruction.operand_2 and instruction.operand_2 > 0:
                    live_values.add(instruction.operand_2)
        return live_values

    def gather_initialized_live_values(blocks: set[BasicBlock]) -> set[int]:
        initialized_live_values = set()
        for block in blocks:
            for instruction in block.get_instructions():
                if instruction.operator in RegisterAllocator.RELEVANT_OPERATORS:
                    initialized_live_values.add(instruction.ssa_value)
        return initialized_live_values
    
    def interfere_all_live_values(live_set: set[int]) -> None:
        for v in live_set:
            for v2 in live_set:
                if v != v2:
                    ig.declare_edge(v, v2)
        return

    def interfere_while_structure(live_set: set[int], blocks: set[BasicBlock]) -> set[int]:
        live_set |= gather_live_values(blocks)
        live_initialized_set = gather_initialized_live_values(blocks)
        interfere_all_live_values(live_set)
        return live_set - live_initialized_set

    def create_interferences(function: Function) -> None:
        basic_blocks = function.basic_blocks.values()
        while_blocks = [bb for bb in basic_blocks if bb.is_while_block and not bb.is_pre_while_block]
        whiles_structs: dict[BasicBlock, set[BasicBlock]] = gather_top_while_structures(while_blocks)
        worklist: Queue[BasicBlock] = Queue()
        worklist.put(function.last_basic_block)
        results: dict[BasicBlock, set[int]] = dict() # basic blocks to live sets
        while not worklist.empty():
            basic_block = worklist.get()
            if not all(child in results or child.is_while_block for child in cf.src_dests_norm.get(basic_block, set())):
                worklist.put(basic_block)
                continue
            merged_live_set = set().union(*[results[child] for child in cf.src_dests_norm.get(basic_block, set()) if not child.is_while_block])
            results[basic_block] = interfere_basic_block(merged_live_set, basic_block)
            if basic_block.is_while_block and not basic_block.is_pre_while_block:
                results[basic_block] = interfere_while_structure(results[basic_block], whiles_structs[basic_block])
            for parent in cf.src_dests_reversed.get(basic_block, set()):
                if parent.is_od_block:
                    continue
                worklist.put(parent)
        return None
        
    def find_basic_block_with_ssa_value(basic_blocks: set[BasicBlock], ssa_value: int) -> BasicBlock:
        for basic_block in basic_blocks:
            if ssa_value in basic_block.variable_names_ssa_values.values():
                return basic_block
        raise ValueError(f"No basic block found with ssa value {ssa_value}")

    def rewrite_program(nodes_registers: dict[int, int]) -> None:
        for function in program.functions.values():
            for basic_block in function.basic_blocks.values():
                for index, instruction in enumerate(basic_block.join_instructions):
                    if instruction.operator is not Operator.PHI:
                        raise TypeError(f"Expected phi instruction, received {str(instruction)}")
                    phi_ssa_value = instruction.ssa_value
                    operand_1_ssa_value = instruction.operand_1
                    operand_2_ssa_value = instruction.operand_2
                    operand_1_regno = nodes_registers.get(operand_1_ssa_value, operand_1_ssa_value)
                    operand_2_regno = nodes_registers.get(operand_2_ssa_value, operand_2_ssa_value)
                    operand_1_regname = RegisterAllocator.regno_to_regname(operand_1_regno)
                    operand_2_regname = RegisterAllocator.regno_to_regname(operand_2_regno)
                    parents = cf.src_dests_reversed.get(basic_block, set())
                    phi_regno = nodes_registers.get(phi_ssa_value, None)
                    basic_block.join_instructions[index] = Instruction(phi_ssa_value, Operator.EMPTY_INSTRUCTION)
                    if phi_regno is None:
                        continue
                    phi_regname = RegisterAllocator.regno_to_regname(phi_regno)
                    if operand_1_regno != phi_regno:
                        target_basic_block = find_basic_block_with_ssa_value(parents, operand_1_ssa_value)
                        target_basic_block.instructions.append(Instruction("∞", Operator.MOV, operand_1_regname, phi_regname))
                    if operand_2_regno != phi_regno:
                        target_basic_block = find_basic_block_with_ssa_value(parents, operand_2_ssa_value)
                        target_basic_block.instructions.append(Instruction("∞", Operator.MOV, operand_2_regname, phi_regname))
                # basic_block.join_instructions = [] # remove phi instructions

                for instruction in basic_block.instructions + basic_block.branch_instructions:
                    if instruction.operator in RegisterAllocator.BRANCH_OPERATORS:
                        instruction.ssa_value = "∞"
                    if instruction.operand_1 and instruction.operand_1 in nodes_registers:
                        instruction.operand_1 = RegisterAllocator.regno_to_regname(nodes_registers[instruction.operand_1])
                    if instruction.operand_2 and instruction.operand_2 in nodes_registers:
                        instruction.operand_2 = RegisterAllocator.regno_to_regname(nodes_registers[instruction.operand_2])
                    if instruction.ssa_value in nodes_registers:
                        instruction.ssa_value = RegisterAllocator.regno_to_regname(nodes_registers[instruction.ssa_value])
        return

    def backward_pass() -> None:
        for function in program.functions.values():
            create_interferences(function)
        nodes_registers = RegisterAllocator.welsh_powell_color(ig.adjacency_list)
        rewrite_program(nodes_registers)
        ig.emit_graph(nodes_registers)
        dot.emit_regalloc_program()
        return

    # Emit forward pass
    file_stream.next_token()
    forward_pass()
    
    # Emit backward pass
    backward_pass()

    # closing all output streams except STDOUT
    close_files(*output_streams)
    

if __name__ == "__main__":
    compile()