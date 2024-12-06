#include "main.hpp"
#include "Dataset.hpp"
/* TODO: Please design your data structure carefully so that you can work with the given dataset
 *       in this assignment. The below structures are just some suggestions.
 */


struct kDTreeNode
{
    vector<int> data;
    kDTreeNode *left;
    kDTreeNode *right;
    int label;
    kDTreeNode(vector<int> data, kDTreeNode *left = nullptr, kDTreeNode *right = nullptr)
    {
        this->data = data;
        this->left = left;
        this->right = right;
        label = -1;
    }
    kDTreeNode(vector<int> data, int label, kDTreeNode *left = nullptr, kDTreeNode *right = nullptr)
    {
        this->data = data;
        this->left = left;
        this->right = right;
        this->label = label;
    }

    friend ostream &operator<<(ostream &os, const kDTreeNode &node)
    {
        os << "(";
        for (int i = 0; i < node.data.size(); i++)
        {
            os << node.data[i];
            if (i != node.data.size() - 1)
            {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }
};

class NDist
{
public:
    double dist;
    kDTreeNode *node;
    NDist(double dist, kDTreeNode *node)
    {
        this->dist = dist;
        this->node = node;
    }
    bool operator<(const NDist &other) const
    {
        return dist < other.dist;
    }
    bool operator>(const NDist &other) const
    {
        return dist > other.dist;
    }
    bool operator<=(const NDist &other) const
    {
        return dist <= other.dist;
    }
};


class kDTree
{
private:
    int k;
    kDTreeNode *root;

public:
    kDTree(int k = 2);//done
    ~kDTree();//done

    const kDTree &operator=(const kDTree &other);//done
    kDTree(const kDTree &other);//done
    void inorderTraversal() const;//done
    void preorderTraversal() const;//done
    void postorderTraversal() const;//done
    int height() const;//done
    int nodeCount() const;//done
    int leafCount() const;//done

    void insert(const vector<int> &point);//done
    void remove(const vector<int> &point);//done
    bool search(const vector<int> &point);//done
    void buildTree(const vector<vector<int>> &pointList);//done
    void nearestNeighbour(const vector<int> &target, kDTreeNode *&best);//done
    void kNearestNeighbour(const vector<int> &target, int k, vector<kDTreeNode *> &bestList);//done

    //Added functions
    void clear(kDTreeNode *node);//done

    /*Traversal*/
    void inorderRecursive(kDTreeNode *node, ostringstream &oss) const;//done
    void preorderRecursive(kDTreeNode *node, ostringstream &oss) const;//done
    void postorderRecursive(kDTreeNode *node, ostringstream &oss) const;//done

    /*Recursion for insert*/
    kDTreeNode* insertRecursive(kDTreeNode *&node, const vector<int> &point, int depth);//done

    /*Recursion for build*/
    void mergeSort(vector<vector<int>> &points, int dim, int low, int high);//done
    void merge(vector<vector<int>> &points, int dim, int low, int mid, int high);//done
    kDTreeNode * insertFromNode(kDTreeNode *node,vector<int>&point, int depth);//done
    kDTreeNode* buildTreeRecursive(kDTreeNode * node,vector<vector<int>> &pointList, int depth);//done

    /*Recursion for removal*/
    kDTreeNode * smallestNode(kDTreeNode *node, int dim, int depth);//done
    kDTreeNode * smallestNodeRecursive(kDTreeNode *node, int dim, int depth);//done
    bool isEqual(const vector<int> &point1, const vector<int> &point2);//done
    kDTreeNode * removeRecursive(kDTreeNode * node, const vector<int> &point, int depth);//done

    /*Recursion for search*/
    bool searchRecursive(kDTreeNode *node, const vector<int> &point, int depth);//done

    /*Recursion for height, nodeCount, leafCount*/
    int heightRecursive(kDTreeNode *node) const;//done
    int nodeCountRecursive(kDTreeNode *node) const;//done
    int leafCountRecursive(kDTreeNode *node) const;//done

    /* Functions for nearest neighbor*/
    double distance(const vector<int> &point1, const vector<int> &point2);//done
    void nearestNeighbourRecursive(kDTreeNode *node, const vector<int> &target, kDTreeNode *&best, double&R, int depth);//done

    /* Recursion for assignment operator*/
    void copyRecursive(kDTreeNode *other);//done

    /*Functions for kNN*/
    kDTreeNode* insertkNN(kDTreeNode * node,const vector<int> &point, int&label, int depth);//done
    void mergeSortkNN(vector<vector<int>> &points, vector<int> &labels, int dim, int low, int high);//done
    void mergekNN(vector<vector<int>> &points, vector<int> &labels, int dim, int low, int mid, int high);//done
    kDTreeNode* buildTreekNNRecursive(kDTreeNode* node, vector<vector<int>> &pointList, vector<int> &labels, int depth);//done
    void buildTreekNN(const vector<vector<int>> &pointList, const vector<int> &labels);//done
    kDTreeNode * findNodeRecursive(kDTreeNode *node, const vector<int> &point, int depth);//done
    kDTreeNode * findNode(const vector<int> &point);//done
    
    /*Functions for k nearest neighbors*/
    void kNearestNeighbourRecursive(kDTreeNode *node, const vector<int> &target, list<NDist> &distList, int k, int depth);//done

};
class kNN
{
private:
    int k;
    kDTree *X_train_tree;
    string label_name;
public:
    kNN(int k = 5);//done
    ~kNN();//done
    void fit(Dataset &X_train, Dataset &y_train);//done
    Dataset predict(Dataset &X_test);//done
    double score(const Dataset &y_test, const Dataset &y_pred);//done
};

// Please add more or modify as needed
