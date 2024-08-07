import platform  # For getting the Python version
import time  # For handling timestamps
import sys  # For accessing command-line arguments
import binascii  # For converting binary data to hexadecimal
import marshal  # For loading the compiled code object from the .pyc file
import dis  # For disassembling the bytecode
import struct  # For unpacking binary data

def write_disassembled_to_file(code, output_path):
    """Write disassembled code to a binary file."""
    # Open the output file in binary write mode
    with open(output_path, 'wb') as output_file:
        # Write the code information to the file
        output_file.write(dis.code_info(code).encode('utf-8'))
        # Write each instruction of the disassembled code to the file
        for instruction in dis.get_instructions(code):
            output_file.write(str(instruction).encode('utf-8') + b'\n')

def view_pyc_file(input_path, output_path):
    """Read and display the content of Python's bytecode in a .pyc file and write to a binary file."""
    
    # Open the .pyc file in binary read mode
    with open(input_path, 'rb') as file:
        magic = file.read(4)  # Read the magic number
        timestamp = file.read(4)  # Read the timestamp
        size = None

        # Check for Python 3.3+ to read the size field
        if sys.version_info.major == 3 and sys.version_info.minor >= 3:
            size = file.read(4)
            size = struct.unpack('I', size)[0]  # Unpack the size field

        code = marshal.load(file)  # Load the compiled code object

        # Convert magic number to hexadecimal
        magic = binascii.hexlify(magic).decode('utf-8')
        # Convert timestamp to human-readable format
        timestamp = time.asctime(time.localtime(struct.unpack('I', timestamp)[0]))

        # Disassemble the bytecode and print it
        dis.disassemble(code)

        # Print a separator and the bytecode information
        print('-' * 80)
        print(
            'Python version: {}\nMagic code: {}\nTimestamp: {}\nSize: {}'
            .format(platform.python_version(), magic, timestamp, size)
        )

        # Write the disassembled code to the output file
        write_disassembled_to_file(code, output_path)

if __name__ == '__main__':
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_pyc_file> <output_bin_file>")
    else:
        input_path = sys.argv[1]  # Get the input file path from the arguments
        output_path = sys.argv[2]  # Get the output file path from the arguments
        # Call the function to process the .pyc file and write the output
        view_pyc_file(input_path, output_path)
