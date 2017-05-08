OBJ = clean ssh

CC = g++

all: $(OBJ)

matmult: ssh.cpp
	$(CC) $(OPT) -o ssh ssh.c

clean:
	/bin/rm -rf $(OBJ) core*

veryclean: clean
	/bin/rn -f *~