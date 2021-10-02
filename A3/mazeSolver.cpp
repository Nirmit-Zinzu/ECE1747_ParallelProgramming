#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <stack>
#include <stdlib.h>
#include <time.h>
#include <chrono>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

using namespace std;

// Code adapted from: https://github.com/zakjwalton/mazeSolver
// g++ mazeSolver.cpp -o run; ./run sample_mazes/20_20.txt

enum Dir
{
	RIGHT = 0,
	LEFT = 1,
	UP = 2,
	DOWN = 3
};

class Coord
{
	int xpos, ypos;
	Dir dir;

public:
	Coord(int x, int y, Dir d)
	{
		xpos = x;
		ypos = y;
		dir = d;
	}
	int getX() { return xpos; }
	int getY() { return ypos; }
	Dir getDir() { return dir; }
	void setX(int x) { xpos = x; }
	void setY(int y) { ypos = y; }
	void setDir(Dir d) { dir = d; }
	void setCoord(int x, int y)
	{
		xpos = x;
		ypos = y;
	}
};

struct maze
{
	int rows;
	int cols;
	char matrix[14001][14001];
};

maze myMaze;

bool is_file_exist(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}

int bruteForceMazeSolver(int i, int j)
{

	int curr_x = j;
	int curr_y = i;
	int count = 0;
	int distance = 0;
	stack<char> moves;

	while (myMaze.matrix[curr_x][curr_y] != 'F')
	{
		//try going right
		if (myMaze.matrix[curr_x][curr_y + 1] == ' ' || myMaze.matrix[curr_x][curr_y + 1] == 'F')
		{
			// cout << "right" << endl;
			count++;
			distance++;
			curr_y++;
			if (myMaze.matrix[curr_x][curr_y] == 'F')
			{
				break;
			}
			moves.push('r');
			myMaze.matrix[curr_x][curr_y] = 'o';
		}

		//try going up
		else if (myMaze.matrix[curr_x - 1][curr_y] == ' ' || myMaze.matrix[curr_x - 1][curr_y] == 'F')
		{
			// cout << "up" << endl;
			count++;
			distance++;
			//break;
			curr_x--;
			if (myMaze.matrix[curr_x][curr_y] == 'F')
			{
				break;
			}
			moves.push('u');
			myMaze.matrix[curr_x][curr_y] = 'o';
		}

		//try going down
		else if (myMaze.matrix[curr_x + 1][curr_y] == ' ' || myMaze.matrix[curr_x + 1][curr_y] == 'F')
		{
			// cout << "down" << endl;
			count++;
			distance++;
			curr_x++;
			if (myMaze.matrix[curr_x][curr_y] == 'F')
			{
				break;
			}
			moves.push('d');
			myMaze.matrix[curr_x][curr_y] = 'o';
		}

		//try going left
		else if (myMaze.matrix[curr_x][curr_y - 1] == ' ' || myMaze.matrix[curr_x][curr_y - 1] == 'F')
		{
			count++;
			distance++;
			// cout << "left" << endl;
			//break;
			curr_y--;
			if (myMaze.matrix[curr_x][curr_y] == 'F')
			{
				break;
			}
			moves.push('l');
			myMaze.matrix[curr_x][curr_y] = 'o';
		}

		//try backtracking right
		else if (moves.top() == 'l')
		{
			moves.pop();
			count++;
			distance--;
			if (myMaze.matrix[curr_x][curr_y] != 'S')
			{
				myMaze.matrix[curr_x][curr_y] = 'x';
			}
			curr_y++;
			// cout << "backtracked right" << endl;
		}

		//try backtracking up
		else if (moves.top() == 'd')
		{
			moves.pop();
			count++;
			distance--;
			if (myMaze.matrix[curr_x][curr_y] != 'S')
			{
				myMaze.matrix[curr_x][curr_y] = 'x';
			}
			curr_x--;
			// cout << "backtracked up" << endl;
		}

		//try backtracking down
		else if (moves.top() == 'u')
		{
			moves.pop();
			count++;
			distance--;
			if (myMaze.matrix[curr_x][curr_y] != 'S')
			{
				myMaze.matrix[curr_x][curr_y] = 'x';
			}
			curr_x++;
			// cout << "backtracked down" << endl;
		}

		//try backtracking left
		else if (moves.top() == 'r')
		{
			moves.pop();
			count++;
			distance--;
			if (myMaze.matrix[curr_x][curr_y] != 'S')
			{
				myMaze.matrix[curr_x][curr_y] = 'x';
			}
			curr_y--;
			// cout << "backtracked left" << endl;
		}

		else
		{
			cout << "Error: Maze Unsolvable" << endl;
			break;
		}

		// if(count > 500){
		// 	break;
		// }
	}

	//Clear X's
	for (int k = 0; k < myMaze.rows; k++)
	{
		for (int h = 0; h < myMaze.cols; h++)
		{
			if (myMaze.matrix[k][h] == 'x')
			{
				myMaze.matrix[k][h] = ' ';
			}
		}
	}

	return distance;
}

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		std::cout << "Usage: " << argv[0] << " <maze_file>\n";
		exit(-1);
	}

	if (!is_file_exist(argv[1]))
	{
		cout << "File location does not exist! Check the input: " << argv[1] << endl;
		exit(-1);
	}

	//required variables
	ifstream in;
	in.open(argv[1]);
	char line;

	//read the matrix using plain c code, character by character
	in >> myMaze.rows;
	in >> line;
	in >> myMaze.cols;
	cout << "Reading a " << myMaze.rows << " by " << myMaze.cols << " matrix." << endl;

	//Burn the end of line character
	in.ignore(200, '\n');
	for (int i = 0; i < myMaze.rows; i++)
	{
		for (int j = 0; j < myMaze.cols; j++)
		{
			in.get(myMaze.matrix[i][j]);
		}
		//Burn the end of line character
		in.ignore(200, '\n');
	}

	// Print the empty maze ------ Comment below for big maze sizes
	// for(int i=0; i<myMaze.rows; i++)
	// {
	// 	for(int j=0; j<myMaze.cols; j++)
	// 		cout << myMaze.matrix[i][j];
	// 	cout << endl;
	// }

	int x = 1, y = 1;

	//Find starting coordinates
	for (int i = 0; i < myMaze.rows; i++)
	{
		for (int j = 0; j < myMaze.cols; j++)
		{
			if (myMaze.matrix[i][j] == 'S')
			{
				x = j;
				y = i;
			}
		}
	}

	cout << "=> Starting: brute force algorithm." << endl;
	auto startTimeBf = std::chrono::steady_clock::now();
	int bfDistance = bruteForceMazeSolver(x, y);
	auto endTimeBf = std::chrono::steady_clock::now();
 	auto usBf = std::chrono::duration_cast<std::chrono::microseconds>(endTimeBf - startTimeBf);

	//Print Maze ------ Comment below for big maze sizes
	// for(int i=0; i<myMaze.rows; i++)
	// {
	// 	for(int j=0; j<myMaze.cols; j++)
	// 		cout << myMaze.matrix[i][j];
	// 	cout << endl;
	// }

	cout << "========== Summary ==========" << endl;
	cout << "Brute Force:" << endl;
	cout << "   Distance (units): " << bfDistance << endl;
	cout << "   Time (us):" << usBf.count() << endl;

	return 0;
}
