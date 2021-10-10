//////////////////////////////////////////////////////////////////////////
// Maze solver - Parallel Solution
// Note: this is a C++ program: compile it with g++
//
// ================= Group Members =================
// Charvi Choksi       | 1006952317
// Shwetha Padmanabhan | 1007541242
// Nirmit Zinzuwadia   | 1002434074
//////////////////////////////////////////////////////////////////////////

// Sequential Code adapted from: https://github.com/zakjwalton/mazeSolver

// To Run:
// g++ mazeSolverParallel.cpp -lpthread -o runParallel; ./runParallel sample_mazes/14001_14001.txt

// Extra notes:
// i = y
// j = x

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

const int MAX_THREADS = 8;

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
	char **matrix;
};

struct slabInfo {
	int yMin, yMax;  // slab range
	int sX, sY;
	int fX, fY;
	int distance = 0;
};

maze myMaze;

bool is_file_exist(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}

bool isWithinSlab(int y, int x, slabInfo *slab) {
	if (y >= slab->yMin && y <=  slab->yMax && x >= 0 && x < myMaze.cols) {
		return true;
	}
	return false;
}

// printing the maze ------ Comment below for big maze sizes
void printMaze() {
	for (int i = 0; i < myMaze.rows; i++)
	{
		for (int j = 0; j < myMaze.cols; j++)
			cout << myMaze.matrix[i][j];
		cout << endl;
	}
}

void* bruteForceMazeSolver(void* arg)
{
	slabInfo *slab = (slabInfo *)arg;
	int curr_x = slab->sX;
	int curr_y = slab->sY;
	int count = 0;
	int distance = 0;
	stack<char> moves;

	while (myMaze.matrix[curr_y][curr_x] != 'F')
	{
		// cout << curr_x << " " << curr_y << endl;
		// printMaze();

		//try going right
		if (isWithinSlab(curr_y, curr_x + 1, slab) && (myMaze.matrix[curr_y][curr_x + 1] == ' ' || myMaze.matrix[curr_y][curr_x + 1] == 'F'))
		{
			// cout << "right" << endl;
			count++;
			distance++;
			curr_x++;
			if (myMaze.matrix[curr_y][curr_x] == 'F')
			{
				break;
			}
			moves.push('r');
			myMaze.matrix[curr_y][curr_x] = 'o';
		}

		//try going up
		else if (isWithinSlab(curr_y - 1, curr_x, slab) && (myMaze.matrix[curr_y - 1][curr_x] == ' ' || myMaze.matrix[curr_y - 1][curr_x] == 'F'))
		{
			// cout << "up" << endl;
			count++;
			distance++;
			//break;
			curr_y--;
			if (myMaze.matrix[curr_y][curr_x] == 'F')
			{
				break;
			}
			moves.push('u');
			myMaze.matrix[curr_y][curr_x] = 'o';
		}

		//try going down
		else if (isWithinSlab(curr_y + 1, curr_x, slab) && (myMaze.matrix[curr_y + 1][curr_x] == ' ' || myMaze.matrix[curr_y + 1][curr_x] == 'F'))
		{
			// cout << "down" << endl;
			count++;
			distance++;
			curr_y++;
			if (myMaze.matrix[curr_y][curr_x] == 'F')
			{
				break;
			}
			moves.push('d');
			myMaze.matrix[curr_y][curr_x] = 'o';
		}

		//try going left
		else if (isWithinSlab(curr_y, curr_x - 1, slab) && (myMaze.matrix[curr_y][curr_x - 1] == ' ' || myMaze.matrix[curr_y][curr_x - 1] == 'F'))
		{
			count++;
			distance++;
			// cout << "left" << endl;
			//break;
			curr_x--;
			if (myMaze.matrix[curr_y][curr_x] == 'F')
			{
				break;
			}
			moves.push('l');
			myMaze.matrix[curr_y][curr_x] = 'o';
		}

		//try backtracking right
		else if (moves.top() == 'l')
		{
			moves.pop();
			count++;
			distance--;
			if (myMaze.matrix[curr_y][curr_x] != 'S')
			{
				myMaze.matrix[curr_y][curr_x] = 'x';
			}
			curr_x++;
			// cout << "backtracked right" << endl;
		}

		//try backtracking up
		else if (moves.top() == 'd')
		{
			moves.pop();
			count++;
			distance--;
			if (myMaze.matrix[curr_y][curr_x] != 'S')
			{
				myMaze.matrix[curr_y][curr_x] = 'x';
			}
			curr_y--;
			// cout << "backtracked up" << endl;
		}

		//try backtracking down
		else if (moves.top() == 'u')
		{
			moves.pop();
			count++;
			distance--;
			if (myMaze.matrix[curr_y][curr_x] != 'S')
			{
				myMaze.matrix[curr_y][curr_x] = 'x';
			}
			curr_y++;
			// cout << "backtracked down" << endl;
		}

		//try backtracking left
		else if (moves.top() == 'r')
		{
			moves.pop();
			count++;
			distance--;
			if (myMaze.matrix[curr_y][curr_x] != 'S')
			{
				myMaze.matrix[curr_y][curr_x] = 'x';
			}
			curr_x--;
			// cout << "backtracked left" << endl;
		}

		else
		{
			cout << "Error: Maze Unsolvable" << endl;
			break;
		}
	}

	//Clear X's
	for (int k = slab->yMin; k <= slab->yMax; k++)
	{
		for (int h = 0; h < myMaze.cols; h++)
		{
			if (myMaze.matrix[k][h] == 'x')
			{
				myMaze.matrix[k][h] = ' ';
			}
		}
	}

	slab->distance = distance;
	return nullptr;
}

int parallelMazeSolver(int x, int y)
{
	slabInfo slabs[MAX_THREADS];
	int increment = myMaze.rows/MAX_THREADS;

	// figure out yMin, yMax
	#pragma omp parallel for
	for (int i = 0; i < MAX_THREADS; i++) {
		slabs[i].yMin = i * increment;
		if (i == MAX_THREADS - 1) {
			slabs[i].yMax = max( ((i+1) * increment) - 1 , myMaze.rows - 1);
		} else {
			slabs[i].yMax = ((i+1) * increment) - 1;
		}
	}

	// figure out sX, sY;
	// assume (i,j) = first quandrant
	slabs[0].sX = x;
	slabs[0].sY = y;
	bool flag = false;
	#pragma omp parallel for
	for (int slab = 1; slab < MAX_THREADS; slab++) {
		#pragma omp parallel for collapse(2)
		for (int i = slabs[slab].yMin; i < myMaze.rows; i++) {
			for (int j = 0; j < myMaze.cols; j++) {
				if(myMaze.matrix[i][j] == ' ' && myMaze.matrix[i-1][j] == ' ') {
					myMaze.matrix[i][j] = 'S';
					slabs[slab].sX = j;
					slabs[slab].sY = i;
					myMaze.matrix[i-1][j] = 'F';
					slabs[slab-1].fX = j;
					slabs[slab-1].fY = i-1;
					flag = true;
					break;
				}
			}
			if (flag) break;
		}
	}

	pthread_t threads[MAX_THREADS];
	for (int i = 0; i < MAX_THREADS; i++) {
		int rc = pthread_create(&threads[i], NULL, bruteForceMazeSolver, (void *)&slabs[i]);
		if (rc) {
			std::cout << "Error:unable to create thread," << rc << std::endl;
			exit(-1);
		}
	}

	int parallelDistance = 0;
	for (int i = 0; i < MAX_THREADS; i++) {
		pthread_join(threads[i], NULL);
		parallelDistance += slabs[i].distance;
	}

	// Instead of using pthread, we can use the following, though this is slower somehow
	// Compier doesn't optimize much using the pragmas here.
	// #pragma omp parallel for reduction(+ : parallelDistance)
	// for (int slab = 0; slab < MAX_THREADS; slab++) {
	// 	cout << "Start x " << slabs[slab].sX << " Start Y " << slabs[slab].sY << endl;
	// 	parallelDistance += bruteForceMazeSolver(slabs[slab]);
	// }

	return parallelDistance;
}

int sequentialMazeSolver(int x, int y)
{
	slabInfo slabs;
	slabs.yMin = 0;
	slabs.yMax = myMaze.rows - 1;
	// start x,y
	slabs.sX = x;
	slabs.sY = y;

	bruteForceMazeSolver((void *)&slabs);

	//Print Maze ------ Comment below for big maze sizes
	// for (int i = 0; i < myMaze.rows; i++)
	// {
	// 	for (int j = 0; j < myMaze.cols; j++)
	// 		cout << myMaze.matrix[i][j];
	// 	cout << endl;
	// }

	return slabs.distance;
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

	// allocate 2D maze array
	myMaze.matrix = new char *[myMaze.rows];
	for (int i = 0; i < myMaze.rows; i++)
	{
		myMaze.matrix[i] = new char[myMaze.cols];
	}

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

	cout << "=> Starting: Sequential algorithm." << endl;
	auto startTimeBf = std::chrono::steady_clock::now();
	int bfDistance = sequentialMazeSolver(x, y);
	auto endTimeBf = std::chrono::steady_clock::now();
	auto usBf = std::chrono::duration_cast<std::chrono::microseconds>(endTimeBf - startTimeBf);

	// clear the maze for parallel run
	for (int i = 0; i < myMaze.rows; i++)
	{
		for (int j = 0; j < myMaze.cols; j++) {
			if (myMaze.matrix[i][j] == 'o') {
				myMaze.matrix[i][j] = ' ';
			}
		}
	}

	cout << "=> Starting parallel algorithm." << endl;
	auto startTimeParallel = std::chrono::steady_clock::now();
	int parallelDistance = parallelMazeSolver(x, y);
	auto endTimeParallel = std::chrono::steady_clock::now();
	auto usParallel = std::chrono::duration_cast<std::chrono::microseconds>(endTimeParallel - startTimeParallel);

	// delete allocated dynamic maze array
	if (myMaze.matrix != nullptr)
	{
		for (int i = 0; i < myMaze.rows; i++)
		{
			delete[] myMaze.matrix[i];
		}
		delete[] myMaze.matrix;
	}

	cout << "========== Summary ==========" << endl;
	cout << "Sequential:" << endl;
	cout << "   Distance (units): " << bfDistance << endl;
	cout << "   Time (us): " << usBf.count() << endl;
	cout << "Parallel Algorithm:" << endl;
	cout << "   Number of threads used: " << MAX_THREADS << endl;
	cout << "   Distance (units): " << parallelDistance << endl;
	cout << "   Time (us): " << usParallel.count() << endl;

	return 0;
}
