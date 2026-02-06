"""
File: Class representing test case code and its components.

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
from __future__ import annotations

from typing import List, Dict, Optional, Final, TypedDict, Generator as GeneratorType
from dataclasses import dataclass
import shutil

from .actor import Actor, ActorID, ActorName, ActorPL, ActorMode
from .instruction import Instruction
from .test_case_binary import TestCaseBinary
from .test_case_binary import SymbolTableEntry, InstructionMap
import copy

from ..logs import error


# ==================================================================================================
# Program Structure: CodeSection -> Function -> BasicBlock -> InstructionNode -> Instruction
# ==================================================================================================
@dataclass
class InstructionNode:
    """
    Wrapper class to represent an instruction as a node
    in a double-linked list that constitutes a basic block
    """
    instruction: Final[Instruction]
    """ Wrapped instruction object """

    parent: Final[BasicBlock]
    """ Basic block to which the instruction belongs """

    next: Optional[InstructionNode] = None
    """ Next instruction in the basic block """

    previous: Optional[InstructionNode] = None
    """ Previous instruction in the basic block """

    def __init__(self, instruction: Instruction, parent: BasicBlock):
        self.instruction = instruction
        self.parent = parent
        self.next = None
        self.previous = None

    def __str__(self) -> str:
        return str(self.instruction)


class BasicBlock:
    """ Basic block in the test case code """

    name: Final[str]
    """ The name (i.e., label) of the basic block """

    parent: Final[Optional[Function]]
    """ The function that owns the basic block """

    successors: List[BasicBlock]
    """ List of basic blocks that are successors of this basic block """

    terminators: List[Instruction]
    """ List of terminator instructions in the basic block """

    is_exit: Final[bool]
    """ Indicates whether the basic block should be treated as a function exit block """

    _start: Optional[InstructionNode] = None
    _end: Optional[InstructionNode] = None

    def __init__(self, name: str, parent: Optional[Function] = None, is_exit: bool = False):
        self.name = name
        self.parent = parent
        self.is_exit = is_exit
        self.successors = []
        self.terminators = []

    def __str__(self) -> str:
        return self.name

    def __len__(self) -> int:
        """ Length of the basic block is the number of instructions in it """
        count = 0
        if self._start:
            node = self._start
            count = 1
            while node.next:
                node = node.next
                count += 1
        return count

    def __iter__(self) -> GeneratorType[Instruction, None, None]:
        """ Default iterator over the instructions in the basic block """
        current_node = self._start
        while current_node:
            yield current_node.instruction
            current_node = current_node.next

    def iter_nodes(self) -> GeneratorType[InstructionNode, None, None]:
        """ Non-default iterator: Iterate over the nodes in the basic block """
        current_node = self._start
        while current_node:
            yield current_node
            current_node = current_node.next

    def get_owner(self) -> Actor:
        """ Get the actor that owns the basic block """
        assert self.parent is not None, "Basic block does not have a parent function"
        return self.parent.parent.owner

    # ----------------------------------------------------------------------------------------------
    # Instruction insertion and deletion
    def insert_after(self, position: Optional[InstructionNode], inst: Instruction) -> None:
        """ Insert an instruction after a given position node in a basic block
        :param position: If not None, the node after which to insert the new instruction;
                         If None, insert at the _end of the basic block
        :param inst: The instruction to insert
        :return: None
        :raises ValueError: If `position` is not found in the basic block
        """
        inst_node = InstructionNode(inst, self)

        # Position is None and the BB is empty: set the start and end to the new instruction
        if position is None and self._end is None:
            self._start = inst_node
            self._end = inst_node
            return

        # Position is None and the BB is not empty: set the position to the end of the BB
        if position is None:
            position = self._end
        assert position is not None

        # Position is not None: ensure that `position` belongs to this BB
        if position.parent != self:
            raise ValueError("`position` not found in the basic block")

        # Insert the new instruction
        next_ = position.next
        position.next = inst_node
        inst_node.previous = position
        if next_:
            inst_node.next = next_
            next_.previous = inst_node
        else:
            self._end = inst_node

    def insert_before(self, position: Optional[InstructionNode], inst: Instruction) -> None:
        """ Insert an instruction before a given position node in a basic block
        :param position: If not None, the node before which to insert the new instruction;
                         If None, insert at the beginning of the basic block
        :param inst: The instruction to insert
        :return: None
        :raises ValueError: If `position` is not found in the basic block
        """
        inst_node = InstructionNode(inst, self)

        # Position is None and the BB is empty: set the start and end to the new instruction
        if position is None and self._start is None:
            self._start = inst_node
            self._end = inst_node
            return

        # Position is None and the BB is not empty: set the position to the start of the BB
        if position is None:
            position = self._start
        assert position is not None

        # Position is not None: ensure that `position` belongs to this BB
        if position.parent != self:
            raise ValueError(f"instruction {position} belongs to {position.parent}, not {self}")

        # Insert the new instruction
        previous = position.previous
        position.previous = inst_node
        inst_node.next = position
        if previous:
            inst_node.previous = previous
            previous.next = inst_node
        else:
            self._start = inst_node

    def delete(self, target: InstructionNode) -> None:
        """
        Delete a node from a basic block
        :param target: The node to delete
        :return: None
        :raises ValueError: If the node does not belong to the basic block
        """
        # Verify that this node indeed belongs to this BB
        if target.parent != self:
            raise ValueError("Error deleting an instruction from a BB; instruction not found")

        # Patch the linked list
        previous = target.previous
        next_ = target.next
        if previous is None and next_ is None:  # the only instruction in BB
            self._end = None
            self._start = None
        elif previous is None:  # the first instruction
            next_.previous = None  # type: ignore
            self._start = next_
        elif next_ is None:  # the last instruction
            previous.next = None
            self._end = previous
        else:  # somewhere in the middle
            previous.next = next_
            next_.previous = previous

    # ----------------------------------------------------------------------------------------------
    # Instruction access
    def get_first(self, exclude_macros: bool = False) -> Optional[InstructionNode]:
        """
        Get the first InstructionNode in the basic block
        :param exclude_macros: If True, return the first non-macro instruction
        :return: The first node or None if the basic block is empty
        """
        if not exclude_macros:
            return self._start if self._start is not None else None

        # Skip macro instructions
        entry_node = self.get_first()
        while entry_node:
            if entry_node.instruction.name != "macro":
                break
            entry_node = entry_node.next
        return entry_node

    def get_last(self) -> Optional[InstructionNode]:
        """ Get the last InstructionNode in the basic block
        :return: The last node or None if the basic block is empty
        """
        return self._end if self._end is not None else None

    def find_instruction_node(self, inst: Instruction) -> Optional[InstructionNode]:
        """
        Find a InstructionNode in the basic block that corresponds to a given instruction
        :param inst: The instruction to find
        :return: The node corresponding to the instruction or None if not found
        """
        for node in self.iter_nodes():
            if node.instruction == inst:
                return node
        return None


class Function:
    """
    Function in the test case code.
    This class is essentially a wrapper around a list of basic blocks, with special features:
    * The basic blocks are ordered by their appearance in the assembly code.
    * The last basic block has special handling: it is assumed to be the exit block of the function,
      and it should contain little-to-no instructions. IMPORTANT: This basic block
      is NOT included when iterating over the basic blocks in the function
      or when calculating its length.
    """

    name: Final[str]
    """ The name of the function; matches the function label in the assembly code """

    parent: Final[CodeSection]
    """ The actor that owns the function"""

    _all_bb: List[BasicBlock]
    """ List of all basic blocks in the function, ordered by their appearance in asm """

    def __init__(self, name: str, parent: CodeSection):
        self.name = name
        self.parent = parent
        exit_bb = BasicBlock(f".exit_{name.removeprefix('.function_')}", parent=self, is_exit=True)
        self._all_bb = [exit_bb]

    def __len__(self) -> int:
        """ Length of the function is the number of basic blocks in it, excluding the exit block """
        return len(self._all_bb[:-1])

    def __iter__(self) -> GeneratorType[BasicBlock, None, None]:
        """ Iterate over the basic blocks in the function, excluding the exit block """
        for bb in self._all_bb[:-1]:
            yield bb

    def __getitem__(self, id_: int) -> BasicBlock:
        """ Get a basic block by its index, excluding the exit block """
        assert len(self._all_bb) > 1, "Function has no non-exit basic blocks"
        non_exit_bbs = self._all_bb[:-1]
        return non_exit_bbs[id_]

    def append(self, bb: BasicBlock) -> None:
        """ Append a basic block to the second-to-last position in the function
          (the last is always exit) """
        exit_bb = self._all_bb.pop()
        self._all_bb.append(bb)
        self._all_bb.append(exit_bb)

    def extend(self, bb_list: List[BasicBlock]) -> None:
        """ Extend the function with a list of basic blocks (added to the end) """
        exit_bb = self._all_bb.pop()
        self._all_bb.extend(bb_list)
        self._all_bb.append(exit_bb)

    def get_first_bb(self) -> BasicBlock:
        """ Get the first basic block in the function.
        If there are no basic blocks, return the default exit block.
        """
        return self._all_bb[0]

    def get_exit_bb(self) -> BasicBlock:
        """ Get the last basic block in the function.
        If there are no basic blocks, return the default exit block.
        """
        exit_ = self._all_bb[-1]
        assert exit_.is_exit, "The last basic block is not marked as an exit block"
        return self._all_bb[-1]

    def get_owner(self) -> Actor:
        """ Get the actor that owns the function """
        return self.parent.owner


class _ELFSectionData(TypedDict):
    """ Data of a section in the ELF file """
    offset: int
    size: int
    id: int


class CodeSection:
    """
    Section in the test case code.
    This class is essentially a wrapper around an ordered list of functions, with special features:
    * The functions are ordered by their appearance in the assembly code.
    """

    name: Final[str]
    """ The name of the section """

    owner: Actor
    """ The actor that owns the section """

    id_: Optional[int] = None
    """ ID of the section; must match the ID in the ELF file """

    _functions: Final[List[Function]]  # List of functions in the section
    _bin_offset: Optional[int] = None  # Offset of the section in the object file
    _bin_size: Optional[int] = None  # Size of the section in the object file

    def __init__(self, owner: Actor):
        self.owner = owner
        self.name = owner.name
        owner.assign_code_section(self)
        self._functions = []

    def __iter__(self) -> GeneratorType[Function, None, None]:
        """ Iterate over the functions in the section """
        for func in self._functions:
            yield func

    def __len__(self) -> int:
        """ Length of the section is the number of functions in it """
        return len(self._functions)

    def __getitem__(self, id_: int) -> Function:
        """ Get a function by its index """
        return self._functions[id_]

    def append(self, func: Function) -> None:
        """ Append a function to the section """
        assert func.name not in [f.name for f in self._functions], \
            f"Function {func.name} already exists in the section"
        self._functions.append(func)

    def assign_elf_data(self, offset: int, size: int, id_: int) -> None:
        """ Assign ELF data to the section """
        assert self._bin_offset is None and self._bin_size is None and self.id_ is None, \
            "ELF data is already assigned"
        self._bin_offset = offset
        self._bin_size = size
        self.id_ = id_

    def get_elf_data(self) -> _ELFSectionData:
        """ Get the ELF data of the section """
        assert self._bin_offset is not None and self._bin_size is not None \
            and self.id_ is not None, "ELF data is not assigned"
        return {"offset": self._bin_offset, "size": self._bin_size, "id": self.id_}


# ==================================================================================================
# All Program Information Combined
# ==================================================================================================
TC_EXIT_LABEL = ".test_case_exit"


class Program:
    asm_path: str = ''
    _obj: Optional[ProgramBinary] = None
    #faulty_pte: PageTableModifier
    start: Optional[Instruction] = None
    end: Optional[Instruction] = None
    length: int
    count: int
    maxCount: int
    address_map: Dict[int, Instruction]
    num_prologue_instructions: int = 1

    _sections: Final[List[CodeSection]]  # List of sections in the test case program
    _actors: Dict[ActorName, Actor]  # Dictionary of actors in the test case program
    _tc_exit_bb: Final[BasicBlock]  # Special basic block labeled that terminates the test case


    def __init__(self, maxCount, asm_path, bin_path):
        self.length = 0
        self.count = 0
        self.maxCount = maxCount
        #self.faulty_pte = PageTableModifier()
        self.address_map = {}
        self.asm_path = asm_path
        self._obj = None

        self._tc_exit_bb = BasicBlock(TC_EXIT_LABEL)
        self._actors = {"main": Actor.create_main()}
        self._sections = [CodeSection(self._actors["main"])]

    def __iter__(self):
        current_instruction = self.start
        while current_instruction:
            if (current_instruction == self.end):
                yield current_instruction
                current_instruction = None
                continue
            yield current_instruction
            current_instruction = current_instruction.next

    def __len__(self):
        return self.length
    
    def append(self, inst: Instruction):
        insert = copy.deepcopy(inst)
        if (self.count >= self.maxCount):
            return 
        if not self.start:
            self.start = insert
            self.end = insert
            self.length += 1
            if not inst.is_instrumentation: self.count += 1
            return
        curr_end = self.end
        curr_end.next = insert
        insert.previous = curr_end
        self.end = insert
        if not inst.is_instrumentation: self.count += 1
        self.length += 1

    def getInd(self, ind):
        i = 0
        curr = self.start
        while (i < ind):
            curr = curr.next
            i += 1
        return curr

    def print(self):
        curr = self.start
        while(curr != self.end):
            print(curr)
            curr = curr.next
        print(curr)

    def assign_obj(self, obj_path: str) -> None:
        """
        Assign an object file generated from the assembly file
        :param obj_path: The path to the object file
        :return: None
        :raises AssertionError: If the object file is already assigned
        """
        assert self._obj is None, "Object file is already assigned"
        self._obj = ProgramBinary(obj_path, self)

    def get_obj(self) -> ProgramBinary:
        """
        Get assigned TestCaseBinary, the container of the object file
        generated from the test case program
        """
        assert self._obj is not None, "Object file is not assigned"
        return self._obj
    
    def get_actors(self, sorted_: bool = False) -> List[Actor]:
        """
        Get a list of actors.
        :param sorted: Whether to sort the actors by ID
        :return: A list of actors
        """
        if sorted_:
            return sorted(self._actors.values(), key=lambda x: x.get_id())
        return list(self._actors.values())
    
    def add_actor_with_section(self, actor: Actor, allow_overwrite: bool = False) -> None:
        """
        Add an actor to the test case and assign it an empty CodeSection.

        If an actor with the same name already exists and `allow_overwrite` is True,
        the new actor will overwrite the existing one.
        Otherwise, an error will be raised.
        :param actor: The actor to add
        :param allow_overwrite: Whether to allow overwriting an existing actor
        :return: None
        :raises ValueError: If the actor already exists in the test case
        """
        if not allow_overwrite and actor.name in self._actors:
            raise ValueError(f"Actor {actor.name} already exists in the test case")

        # Update of the main actor
        if actor.is_main:
            assert actor.mode == ActorMode.HOST
            assert actor.privilege_level == ActorPL.KERNEL
            self._actors[actor.name] = actor
            section = self._sections[0]
            section.owner = actor
            actor.assign_code_section(section)
            return

        # Update of an actor
        if allow_overwrite and actor.name in self._actors:
            self._actors[actor.name] = actor
            section = self.find_section(actor.name)
            section.owner = actor
            actor.assign_code_section(section)
            return

        # New actor
        self._actors[actor.name] = actor
        section = CodeSection(actor)
        self._sections.append(section)

    def find_actor(self,
                   name: Optional[ActorName] = None,
                   actor_id: Optional[ActorID] = None) -> Actor:
        """
        Select an actor by name or ID.
        :param name: The name of the actor
        :param actor_id: The ID of the actor
        :return: The actor
        :raises KeyError: If an actor with the given name/ID does not exist in the test case
        :raises ValueError: If neither name nor ID is provided or if both are provided
        """
        # check interface
        assert name is not None or actor_id is not None, "Either name or ID must be provided"
        assert name is None or actor_id is None, "Only one of name or ID should be provided"

        # select by name
        if name is not None:
            if name not in self._actors:
                raise KeyError(f"Actor {name} does not exist in the test case")
            return self._actors[name]

        # select by ID
        for actor in self._actors.values():
            if actor.get_id() == actor_id:
                return actor
        raise KeyError(f"Actor with ID {actor_id} does not exist in the test case")

    def n_actors(self) -> int:
        """
        Get the number of actors in the test case.
        :return: The number of actors
        """
        return len(self._actors)

    # ==============================================================================================
    # Function and section management
    def get_sections(self) -> List[CodeSection]:
        """ Get a list of sections in the test case """
        return self._sections

    def find_section(self, name: str) -> CodeSection:
        """
        Get a section by name
        :param name: The name of the section
        :return: The section
        :raises KeyError: If the section does not exist in the test case
        """
        for sec in self._sections:
            if sec.name == name:
                return sec
        raise KeyError(f"Section {name} does not exist in the test case")

    def find_function(self, name: str) -> Function:
        """
        Get a function by name
        :param name: The name of the function
        :return: The function
        :raises KeyError: If the function does not exist in the test case
        """
        for sec in self._sections:
            for func in sec:
                if func.name == name:
                    return func
        raise KeyError(f"Function {name} does not exist in the test case")

class ProgramBinary:

    obj_path: Final[str]
    _symbol_table: Final[List[SymbolTableEntry]] = None
    _instruction_map: Final[InstructionMap] = None

    _parent: Program


    def __init__(self, obj_path: str, parent: Program):
        self.obj_path = obj_path
        self._parent = parent


    def to_bytes(self, padded_section_size: int = 0, padding_byte: bytes = b'') -> bytes:
        """ Return the full binary of the assembled object file, with sections ordered by actor ID.
        Optionally, pad each section to a specified size with a specified padding byte.

        :param pad_to_size: The size to pad each section to
        :param padding_byte: The byte to use for padding
        :return: A list of byte strings, each containing the full compiled binary of a section
        """

        assert padded_section_size == 0 or len(padding_byte) == 1, \
            "padding_byte must be specified as a single byte if pad_to_size is set"

        code = b''
        with open(self.obj_path, 'rb') as bin_file:
            for actor in self._parent.get_actors(sorted_=True):

                # Read the section from the object file
                section_data = actor.code_section().get_elf_data()
                offset = section_data["offset"]
                size = section_data["size"]

                bin_file.seek(offset)
                code += bin_file.read(size)

                # Apply padding
                assert padded_section_size >= size, \
                    "Padded section size is less than to the original section size"
                if padded_section_size > size:
                    padding = padded_section_size - size
                    code += padding_byte * padding

        return code
    

    def symbol_table(self) -> List[SymbolTableEntry]:
        """ Return the symbol table of the test case program """
        assert self._symbol_table is not None, "Symbol table has not been populated"
        return self._symbol_table
    
    def assign_elf_data(self, symbol_table: List[SymbolTableEntry],
                        instruction_map: InstructionMap) -> None:
        """
        Assign the symbol table and instruction map based on the data parsed from the ELF file
        (normally assigned by an ELFParser instance).
        """
        assert self._symbol_table is None, "Attempting to reassign symbol table"
        assert self._instruction_map is None, "Attempting to reassign instruction map"
        self._symbol_table = symbol_table
        self._instruction_map = instruction_map


    def save_rcbf_program_binary(self, path: str) -> None:
        """
        Save the test case binary in the RCBF format
        (see docs/devel/binary-formats.md for details).
        :param path: The path to save the RCBF file to
        """
        actors = self._parent.get_actors(sorted_=True)
        symbol_table = self.symbol_table()

        # sanity check
        if any(symbol.type_ < 0 for symbol in symbol_table):
            error("attempt to use template as a test case")

        # write the RCBF file
        with open(path, 'wb') as f:
            # header
            f.write((len(actors)).to_bytes(8, byteorder='little'))  # n_actors
            f.write((len(symbol_table)).to_bytes(8, byteorder='little'))  # n_symbols

            # actor metadata
            for actor in actors:
                f.write((actor.get_id()).to_bytes(8, byteorder='little'))
                f.write((actor.mode.value).to_bytes(8, byteorder='little'))
                f.write((actor.privilege_level.value).to_bytes(8, byteorder='little'))
                f.write((actor.data_properties).to_bytes(8, byteorder='little'))
                f.write((actor.data_ept_properties).to_bytes(8, byteorder='little'))
                f.write((0).to_bytes(8, byteorder='little'))  # unused

            # symbol table (first functions sorted by argument, then macros sorted by actor+offset)
            function_symbols = [s for s in symbol_table if s[2] == 0]
            macro_symbols = [s for s in symbol_table if s[2] != 0]
            for aid, s_offset, s_id, arg in sorted(function_symbols, key=lambda s: s.arg):
                # print("function", s_id, aid, s_offset, arg)
                f.write((aid).to_bytes(8, byteorder='little'))
                f.write((s_offset).to_bytes(8, byteorder='little'))
                f.write((s_id).to_bytes(8, byteorder='little'))
                f.write((arg).to_bytes(8, byteorder='little'))
            for aid, s_offset, s_id, arg in sorted(macro_symbols, key=lambda s: (s.sid, s.offset)):
                # print("macro", aid, s_offset, s_id, arg)
                f.write((aid).to_bytes(8, byteorder='little'))
                f.write((s_offset).to_bytes(8, byteorder='little'))
                f.write((s_id).to_bytes(8, byteorder='little'))
                f.write((arg).to_bytes(8, byteorder='little'))

            # section metadata
            for actor in actors:
                section_data = actor.code_section().get_elf_data()
                # print("section\n")
                f.write((section_data["id"]).to_bytes(8, byteorder='little'))
                f.write((section_data["size"]).to_bytes(8, byteorder='little'))
                f.write((0).to_bytes(8, byteorder='little'))

            # code
            with open(self.obj_path, 'rb') as bin_file:
                for actor in actors:
                    section_data = actor.code_section().get_elf_data()
                    bin_file.seek(section_data["offset"])  # type: ignore
                    # print(code, section.size)
                    f.write(bin_file.read(section_data["size"]))

            # print(self.obj_path, f.tell())


class TestCaseProgram:
    """ A class representing a test case program """

    generator_seed: int
    """ Seed used to generate the test case program """

    _asm_path: str  # Path to the assembly file containing the test case program
    _obj: Optional[TestCaseBinary] = None  # Representation of the assembled test case program
    _obj_is_assembled: bool = False  # Flag indicating whether the object file has been assembled

    _sections: Final[List[CodeSection]]  # List of sections in the test case program
    _actors: Dict[ActorName, Actor]  # Dictionary of actors in the test case program
    _tc_exit_bb: Final[BasicBlock]  # Special basic block labeled that terminates the test case
    index: int = 0
    num_prologue_instructions: int = 5 # Number of prologue instructions (hardcoded for now)
    address_map: Dict[int, Instruction]

    def __init__(self, asm_path: str, seed: int = 0):
        self.generator_seed = seed
        self._asm_path = asm_path
        self._tc_exit_bb = BasicBlock(TC_EXIT_LABEL)

        self._actors = {"main": Actor.create_main()}
        self._sections = [CodeSection(self._actors["main"])]
        self.index = 0

    def __len__(self) -> int:
        """ Length of the test case is the number of sections """
        return len(self._sections)

    def __getitem__(self, id_: int) -> CodeSection:
        """ Get a section by its index """
        return self._sections[id_]

    def get_tc_exit_bb(self) -> BasicBlock:
        """ Get the special basic block used to terminate the test case """
        return self._tc_exit_bb

    # ----------------------------------------------------------------------------------------------
    # Iterators
    def __iter__(self) -> GeneratorType[CodeSection, None, None]:
        """ Default iterator over the sections in the test case """
        for sec in self._sections:
            yield sec

    def iter_functions(self) -> GeneratorType[Function, None, None]:
        """ Non-default iterator: Iterate over all functions in the test case """
        for sec in self._sections:
            for func in sec:
                yield func

    def iter_basic_blocks(self) -> GeneratorType[BasicBlock, None, None]:
        """
        Non-default iterator:
        Iterate over all basic blocks in the test case in their order of appearance in the asm file
        """
        for sec in self._sections:
            for func in sec:
                for bb in func:
                    yield bb

    # ----------------------------------------------------------------------------------------------
    # ELF file management
    def assign_obj(self, obj_path: str) -> None:
        """
        Assign an object file generated from the assembly file
        :param obj_path: The path to the object file
        :return: None
        :raises AssertionError: If the object file is already assigned
        """
        assert self._obj is None, "Object file is already assigned"
        self._obj = TestCaseBinary(obj_path, self)

    def mark_as_assembled(self) -> None:
        """ Mark the object file as assembled """
        assert self._obj is not None, "Object file is not assigned"
        self._obj_is_assembled = True
        self._obj.mark_as_assembled()

    def get_obj(self) -> TestCaseBinary:
        """
        Get assigned TestCaseBinary, the container of the object file
        generated from the test case program
        """
        assert self._obj is not None, "Object file is not assigned"
        return self._obj

    # ----------------------------------------------------------------------------------------------
    # ASM file management
    def reassign_asm_file(self, asm_path: str) -> None:
        """ Assign a new assembly file to the test case """
        assert not self._obj_is_assembled, \
            "Attempting to reassign the asm file after it has been assembled"
        self._asm_path = asm_path

    def asm_path(self) -> str:
        """ Get the path to the assigned assembly file """
        return self._asm_path

    def save(self, path: str) -> None:
        """
        Save the test case assembly into a file.
        :param path: The path to the file
        :return: None
        """
        shutil.copy2(self._asm_path, path)

    # ----------------------------------------------------------------------------------------------
    # Actor list management
    def add_actor_with_section(self, actor: Actor, allow_overwrite: bool = False) -> None:
        """
        Add an actor to the test case and assign it an empty CodeSection.

        If an actor with the same name already exists and `allow_overwrite` is True,
        the new actor will overwrite the existing one.
        Otherwise, an error will be raised.
        :param actor: The actor to add
        :param allow_overwrite: Whether to allow overwriting an existing actor
        :return: None
        :raises ValueError: If the actor already exists in the test case
        """
        if not allow_overwrite and actor.name in self._actors:
            raise ValueError(f"Actor {actor.name} already exists in the test case")

        # Update of the main actor
        if actor.is_main:
            assert actor.mode == ActorMode.HOST
            assert actor.privilege_level == ActorPL.KERNEL
            self._actors[actor.name] = actor
            section = self._sections[0]
            section.owner = actor
            actor.assign_code_section(section)
            return

        # Update of an actor
        if allow_overwrite and actor.name in self._actors:
            self._actors[actor.name] = actor
            section = self.find_section(actor.name)
            section.owner = actor
            actor.assign_code_section(section)
            return

        # New actor
        self._actors[actor.name] = actor
        section = CodeSection(actor)
        self._sections.append(section)

    def get_actors(self, sorted_: bool = False) -> List[Actor]:
        """
        Get a list of actors.
        :param sorted: Whether to sort the actors by ID
        :return: A list of actors
        """
        if sorted_:
            return sorted(self._actors.values(), key=lambda x: x.get_id())
        return list(self._actors.values())

    def find_actor(self,
                   name: Optional[ActorName] = None,
                   actor_id: Optional[ActorID] = None) -> Actor:
        """
        Select an actor by name or ID.
        :param name: The name of the actor
        :param actor_id: The ID of the actor
        :return: The actor
        :raises KeyError: If an actor with the given name/ID does not exist in the test case
        :raises ValueError: If neither name nor ID is provided or if both are provided
        """
        # check interface
        assert name is not None or actor_id is not None, "Either name or ID must be provided"
        assert name is None or actor_id is None, "Only one of name or ID should be provided"

        # select by name
        if name is not None:
            if name not in self._actors:
                raise KeyError(f"Actor {name} does not exist in the test case")
            return self._actors[name]

        # select by ID
        for actor in self._actors.values():
            if actor.get_id() == actor_id:
                return actor
        raise KeyError(f"Actor with ID {actor_id} does not exist in the test case")

    def n_actors(self) -> int:
        """
        Get the number of actors in the test case.
        :return: The number of actors
        """
        return len(self._actors)

    # ==============================================================================================
    # Function and section management
    def get_sections(self) -> List[CodeSection]:
        """ Get a list of sections in the test case """
        return self._sections

    def find_section(self, name: str) -> CodeSection:
        """
        Get a section by name
        :param name: The name of the section
        :return: The section
        :raises KeyError: If the section does not exist in the test case
        """
        for sec in self._sections:
            if sec.name == name:
                return sec
        raise KeyError(f"Section {name} does not exist in the test case")

    def find_function(self, name: str) -> Function:
        """
        Get a function by name
        :param name: The name of the function
        :return: The function
        :raises KeyError: If the function does not exist in the test case
        """
        for sec in self._sections:
            for func in sec:
                if func.name == name:
                    return func
        raise KeyError(f"Function {name} does not exist in the test case")


