import platform
import time
import sys
import binascii
import marshal
import dis
import struct

def view_pyc_file(path):
    """Read and display the content of Python's bytecode in a .pyc file."""

    file = open(path, 'rb')  # Open the file in binary read mode

    magic = file.read(4)  # Read the magic number
    timestamp = file.read(4)  # Read the timestamp
    size = None

    # Python 3.3+ has an extra size field
    if sys.version_info.major == 3 and sys.version_info.minor >= 3:
        size = file.read(4)
        size = struct.unpack('I', size)[0]  # Unpack the size field

    code = marshal.load(file)  # Load the compiled code object

    magic = binascii.hexlify(magic).decode('utf-8')  # Convert magic number to hex
    timestamp = time.asctime(time.localtime(struct.unpack('I', timestamp)[0]))  # Convert timestamp to human-readable format

    dis.disassemble(code)  # Disassemble the bytecode

    print('-' * 80)
    print(
        'Python version: {}\nMagic code: {}\nTimestamp: {}\nSize: {}'
        .format(platform.python_version(), magic, timestamp, size)
    )

    file.close()  # Close the file
    
if __name__ == '__main__':
    view_pyc_file(sys.argv[1])
