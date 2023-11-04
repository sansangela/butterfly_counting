#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <cstdint>

using namespace std;

// Define Cmp
struct Cmp {
    bool operator()(const pair<uint64_t, uint64_t>& lhs, const pair<uint64_t, uint64_t>& rhs) const {
        return lhs < rhs;
    }
};

uint64_t read_edge_list_CSR (std::string const &pathname,
			vector<uint64_t>    &row_indices,
			vector<uint64_t>    &col_indices,
			vector<uint64_t> &IA,
			vector<uint64_t> &JA
			)
{
    std::ifstream infile(pathname);
    uint64_t max_id = 0;
    uint64_t num_rows = 0;
    uint64_t src, dst;

    set<pair<uint64_t, uint64_t>, Cmp> s;
    
    while (true)
    {
        infile >> src >> dst;
	      if (infile.eof()) break;

        max_id = std::max(max_id, src);
        max_id = std::max(max_id, dst);

        row_indices.push_back(src);
        col_indices.push_back(dst);

	      s.insert(pair<uint64_t, uint64_t>(src, dst));
	
        ++num_rows;
    }
    std::cout << "%% Read " << num_rows << " rows." << std::endl;
    std::cout << "%% #Nodes = " << (max_id + 1) << std::endl;

    std::cout<< "%% Set size "<<s.size()<<endl;

    IA.push_back(0);

    auto ia_itr = IA.begin();
    uint64_t cur = 0;
    for (auto itr = s.begin(); itr != s.end(); ++itr){
      uint64_t src = (*itr).first;
      uint64_t dst = (*itr).second;
      
      if (cur != src){
        while (cur < src){
          IA.push_back(JA.size());
          cur += 1;
        }
      }
      
      
      JA.push_back(dst);     
    }
    IA.push_back(JA.size());
    

    cout<<"%% IA size:"<<IA.size()<<endl;
    cout<<"%% JA size:"<<JA.size()<<endl;    

    
    return (max_id + 1);
}


int main()
{
    vector<uint64_t> row_indices;
    vector<uint64_t> col_indices;
    vector<uint64_t> IA;
    vector<uint64_t> JA;

    string inputFilePath = "input.txt"; 

    uint64_t maxId = read_edge_list_CSR(inputFilePath, row_indices, col_indices, IA, JA);

    // Print the results or perform any other testing as needed
    cout << "Maximum ID: " << maxId << endl;

    cout << "Row Indices:" << endl;
    for (const auto& row : row_indices) {
        cout << row << " ";
    }
    cout << endl;

    cout << "Column Indices:" << endl;
    for (const auto& col : col_indices) {
        cout << col << " ";
    }
    cout << endl;

    cout << "IA Vector:" << endl;
    for (const auto& ia : IA) {
        cout << ia << " ";
    }
    cout << endl;

    cout << "JA Vector:" << endl;
    for (const auto& ja : JA) {
        cout << ja << " ";
    }
    cout << endl;

    return 0;
}