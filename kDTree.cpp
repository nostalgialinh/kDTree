#include "kDTree.hpp"

/* TODO: You can implement methods, functions that support your data structures here.
 * */

kDTree::kDTree(int k){
    this->k = k;
    this->root = nullptr;
}
void kDTree::clear(kDTreeNode *node){
    if(node != nullptr){
        clear(node->left);
        clear(node->right);
        delete node;
    }
}
kDTree::~kDTree(){
    clear(root);
    root = nullptr;
}
void kDTree::copyRecursive(kDTreeNode *other){
    if(other != nullptr){
        insert(other->data);
        copyRecursive(other->left);
        copyRecursive(other->right);
    }
}
const kDTree &kDTree::operator=(const kDTree &other){
    if(this == &other){
        return *this;
    }
    this->~kDTree();
    this->k = other.k;
    this->copyRecursive(other.root);
    return *this;
}

kDTree::kDTree(const kDTree &other){
    this->k = other.k;
    this->copyRecursive(other.root);
}
void kDTree::inorderRecursive(kDTreeNode *node, std::ostringstream &oss) const {
    if (node != nullptr) {
        inorderRecursive(node->left, oss);
        oss << *node << " ";
        inorderRecursive(node->right, oss);
    }
}

void kDTree::inorderTraversal() const {
    std::ostringstream oss;
    inorderRecursive(root, oss);
    std::string result = oss.str();
    if (!result.empty()) {
        result.pop_back(); // Remove trailing space
    }
    cout << result;
}
void kDTree::preorderRecursive(kDTreeNode *node, ostringstream &oss) const {
    // root left right
    if (node != nullptr) {
        oss << *node << " ";
        preorderRecursive(node->left, oss);
        preorderRecursive(node->right, oss);
    }
}

void kDTree::preorderTraversal() const {
    std::ostringstream oss;
    preorderRecursive(root, oss);
    std::string result = oss.str();
    if (!result.empty()) {
        result.pop_back(); // Remove trailing space
    }
    cout << result;
}

void kDTree::postorderRecursive(kDTreeNode *node, ostringstream &oss) const {
    // left right root
    if (node != nullptr) {
        postorderRecursive(node->left, oss);
        postorderRecursive(node->right, oss);
        oss << *node << " ";
    }
}

void kDTree::postorderTraversal() const {
    std::ostringstream oss;
    postorderRecursive(root, oss);
    std::string result = oss.str();
    if (!result.empty()) {
        result.pop_back(); // Remove trailing space
    }
    cout << result;
}

int kDTree::heightRecursive(kDTreeNode *node) const{
    if(node == nullptr){
        return 0;
    }
    int lheight = heightRecursive(node->left);
    int rheight = heightRecursive(node->right);
    int max = lheight > rheight ? lheight : rheight;
    return max + 1;
}
int kDTree::height() const{
    return heightRecursive(root);
}

int kDTree::nodeCountRecursive(kDTreeNode *node) const{
    if(node == nullptr){
        return 0;
    }
    return 1 + nodeCountRecursive(node->left) + nodeCountRecursive(node->right);
}

int kDTree::nodeCount() const{
    return nodeCountRecursive(root);
}
int kDTree::leafCountRecursive(kDTreeNode *node) const{
    if(node == nullptr){
        return 0;
    }
    if(node->left == nullptr && node->right == nullptr){
        return 1;
    }
    return leafCountRecursive(node->left) + leafCountRecursive(node->right);
}
int kDTree::leafCount() const{
    return leafCountRecursive(root);
}

kDTreeNode* kDTree::insertRecursive(kDTreeNode *&node, const vector<int> &point, int depth){
    /* traveled to leaf*/
    if(node == nullptr){
        return new kDTreeNode(point);
    }


    /* not yet at leaf*/
    int cur_dim, data_tree, data_insert;
    
    cur_dim = depth % this->k;
    data_tree = node->data[cur_dim];
    data_insert = point[cur_dim];
    
    if(data_tree > data_insert){
        /* move to the left*/
        node->left = insertRecursive(node->left,point,depth+1);
    }
    else {
        /*move to the right*/
        node->right = insertRecursive(node->right,point,depth+1);
    }

    return node;
}

void kDTree::insert(const vector<int> &point){
    int sz = (int) point.size();
    /*ensure that data passed to insertRecursive is in correct format*/
    if(sz != this->k){
        return;}
    root = insertRecursive(root,point,0);
}

kDTreeNode * kDTree::smallestNodeRecursive(kDTreeNode *node, int dim, int depth){
    if(node == nullptr){
        return nullptr;
    }
    int current_dim = depth % this->k;
    if(current_dim == dim){
#ifdef SMALL_DBG
        cout << "Current Node1: " << *node << endl;
#endif
        if(node->left == nullptr){
#ifdef SMALL_DBG
            cout << "Returning Node: " << *node << endl;
#endif
            return node;
        }
        else {
            return smallestNodeRecursive(node->left,dim,depth+1);
        }
    }
    else{
#ifdef SMALL_DBG
        cout << "Current Node2: " << *node << endl;
#endif
        kDTreeNode *left = smallestNodeRecursive(node->left,dim,depth+1);
        kDTreeNode *right = smallestNodeRecursive(node->right,dim,depth+1);
        int data_left , data_right, data_node;
        data_left = data_right = data_node = node->data[dim];
        if(left != nullptr){
#ifdef SMALL_DBG
            cout << "Left is not null\n";
#endif
            data_left = left->data[dim];
        }
        if(right != nullptr){
#ifdef SMALL_DBG
            cout << "Right is not null\n";
#endif
            data_right = right->data[dim];
        }
        if(data_node <= data_left && data_node <= data_right){
#ifdef SMALL_DBG
            cout << "Returning Node: " << *node << endl;
#endif
            return node;
        }
        else if(data_left < data_node && data_left < data_right && left != nullptr){
#ifdef SMALL_DBG
            cout << "Returning Left: " << *left << endl;
#endif
            return left;
        }
        else if(right != nullptr){
#ifdef SMALL_DBG
            cout << "Returning Right: " << *right << endl;
#endif
            return right;
        }
    }
#ifdef SMALL_DBG
    cout << "Something went wrong\n";
#endif
    return nullptr;
}

kDTreeNode * kDTree::smallestNode(kDTreeNode *node, int dim, int depth){
    if(node == nullptr){
        return nullptr;
    }
    return smallestNodeRecursive(node,dim,depth);
}
bool kDTree::isEqual(const vector<int> &point1, const vector<int> &point2){
    int sz1 = (int)point1.size();
    int sz2 = (int)point2.size();
    if(sz1 != sz2){
#ifdef SMALL_DBG
        cout << "Sizes are not equal\n";
#endif
        return 0;
    }
    for(int i = 0; i < sz1; i++){
        if(point1[i] != point2[i]){
            return 0;
        }
    }
    return 1;
}
kDTreeNode * kDTree::removeRecursive(kDTreeNode * node, const vector<int> &point, int depth){
    if(node == nullptr){
        return nullptr;
    }

    int cur_dim = depth % this->k;
    if(isEqual(node->data,point)){
        /*found node to delete*/
        if(node->right == nullptr && node->left == nullptr){
            delete node;
            return nullptr;
        }

        if(node->right != nullptr){
            kDTreeNode *smallest = smallestNode(node->right,cur_dim,depth+1);
            node->data = smallest->data;
            node->right = removeRecursive(node->right,smallest->data,depth+1);
        }
        else{
            /*only left substree exists*/
            kDTreeNode *smallest = smallestNode(node->left,cur_dim,depth+1);
            node->data = smallest->data;
            node->right = removeRecursive(node->left,smallest->data,depth+1);
            node->left = nullptr;
        }

        return node;
    }
    if(node->data[cur_dim] > point[cur_dim]){
        node->left = removeRecursive(node->left,point,depth+1);
    }
    else{
        node->right = removeRecursive(node->right,point,depth+1);
    }
    return node;
}

void kDTree::remove(const vector<int> &point){
    root = removeRecursive(root,point,0);
}
bool kDTree::searchRecursive(kDTreeNode *node, const vector<int> &point, int depth){
    if(node == nullptr){
        return 0;
    }
    if(isEqual(node->data,point)){
        return 1;
    }
    int cur_dim = depth % this->k;
    if(node->data[cur_dim] > point[cur_dim]){
        return searchRecursive(node->left,point,depth+1);
    }
    else{
        return searchRecursive(node->right,point,depth+1);
    }
    return 0;
}
bool kDTree::search(const vector<int> &point){
    return searchRecursive(root,point,0);
}

void kDTree::merge(vector<vector<int>> &pointList, int dim, int low, int mid, int high){
    //mid + 1 because mid is included in the left vector
    vector<vector<int>> left(pointList.begin() + low, pointList.begin() + mid+1);
    vector<vector<int>> right(pointList.begin() + mid + 1, pointList.begin() + high + 1);

    int i = 0, j = 0, k = low;
    while (i < left.size() && j < right.size()) {
        if (left[i][dim] <= right[j][dim]) {
            pointList[k++] = left[i++];
        } else {
            pointList[k++] = right[j++];
        }
    }
    while (i < left.size()) {
        pointList[k++] = left[i++];
    }
    while (j < right.size()) {
        pointList[k++] = right[j++];
    }
}

void kDTree::mergeSort(vector<vector<int>> &pointList, int dim, int low, int high){
    if (low < high) {
        int mid = low + (high - low) / 2;
        mergeSort(pointList, dim, low, mid);
        mergeSort(pointList, dim, mid + 1, high);
        merge(pointList, dim, low, mid, high);
    }
}
kDTreeNode* kDTree::insertFromNode(kDTreeNode *node,vector<int> & point, int depth){
    if(node == nullptr){
        return new kDTreeNode(point);
    }
    int cur_dim = depth % this->k;
    if(node->data[cur_dim] > point[cur_dim]){
        return insertFromNode(node->left,point,depth+1);
    }
    return insertFromNode(node->right,point,depth+1);
}
kDTreeNode* kDTree::buildTreeRecursive(kDTreeNode*node,vector<vector<int>> &pointList, int depth){
    if(pointList.empty()){
        return nullptr;
    }
    int sz = (int)pointList.size();
    /*tested with odd and even, obeys policy*/
    int dim = depth % this->k;
    mergeSort(pointList,dim,0,sz-1);

    int median = (sz - 1) / 2;

    /* Median is not included since it is added as the new root (for the recursion)*/
    /* Add in preorder style: root left cright*/
    vector<vector<int>> left(pointList.begin(),pointList.begin() + median);
    vector<vector<int>> right(pointList.begin() + median + 1,pointList.end());

    kDTreeNode * newNode = insertFromNode(node,pointList[median],depth);
    newNode->left= buildTreeRecursive(newNode,left,depth+1);
    newNode->right = buildTreeRecursive(newNode,right,depth+1);
    return newNode;
}

void kDTree::buildTree(const vector<vector<int>>& points) {
    vector<vector<int>> pointsCopy = points;
    root = buildTreeRecursive(root, pointsCopy, 0);
}

double kDTree::distance(const vector<int> &point1, const vector<int> &point2){
    int sz = (int)point1.size();
    double sum = 0;
    for(int i = 0; i < sz; i++){
        sum += pow(point1[i] - point2[i],2);
    }
    return sqrt(sum);
}

void kDTree::nearestNeighbourRecursive(kDTreeNode *node, const vector<int> &target, kDTreeNode *&best,double&R, int depth){
    if(node == nullptr){
        return;
    }
    double dist = distance(node->data,target);
    if(node->left == nullptr && node->right == nullptr){
        if(dist < R){
            R = dist;
            best = node;
        }
        return;
    }
    int cur_dim = depth % this->k;
    int data_tree = node->data[cur_dim];
    int data_target = target[cur_dim];
    double r = (double) abs(data_tree - data_target);

    if(data_target < data_tree){
        /*move to the left*/
        nearestNeighbourRecursive(node->left,target,best,R,depth+1);
        if(r <= R){
            nearestNeighbourRecursive(node->right,target,best,R,depth+1);
        }
    }
    else{
        /*move to the right*/
        nearestNeighbourRecursive(node->right,target,best,R,depth+1);
        if(r <= R){
            nearestNeighbourRecursive(node->left,target,best,R,depth+1);
        }
    }

    if(dist < R){
        R = dist;
        best = node;
    }

}
void kDTree::nearestNeighbour(const vector<int> &target, kDTreeNode *&best){
    double R = 1e9;
    nearestNeighbourRecursive(root,target,best,R,0);
}

kDTreeNode * kDTree::findNodeRecursive(kDTreeNode *node, const vector<int>&point, int depth){
    if(node == nullptr){
        return nullptr;
    }
    if(isEqual(node->data,point)){
        return node;
    }
    int cur_dim = depth % this->k;
    if(node->data[cur_dim] > point[cur_dim]){
        return findNodeRecursive(node->left,point,depth+1);
    }
    else{
        return findNodeRecursive(node->right,point,depth+1);
    }
}

kDTreeNode * kDTree::findNode(const vector<int>&point){
    return findNodeRecursive(root,point,0);
}

void kDTree::kNearestNeighbour(const vector<int> &target, int k, vector<kDTreeNode *> &bestList) {
    list<NDist> distList;
    kNearestNeighbourRecursive(root, target, distList, k, 0);

    list<NDist>::iterator it = distList.begin();
    while (it != distList.end()){
        bestList.push_back((*it).node);
        ++it;
    }
}

void kDTree::kNearestNeighbourRecursive(kDTreeNode *node, const vector<int> &target, list<NDist> &distList, int k, int depth) {
    if (node == nullptr) {
        return;
    }

    double dist = distance(node->data, target);
    list<NDist>::iterator it = distList.begin();
    while (it != distList.end() && it->dist < dist) {
        ++it;
    }

    distList.insert(it, NDist(dist, node));

    if (distList.size() > k) {
        distList.pop_back();
    }

    int cur_dim = depth % k;
    kDTreeNode *next_branch = nullptr;
    kDTreeNode *opposite_branch = nullptr;

    if (target[cur_dim] < node->data[cur_dim]) {
        next_branch = node->left;
        opposite_branch = node->right;
    } else {
        next_branch = node->right;
        opposite_branch = node->left;
    }

    kNearestNeighbourRecursive(next_branch, target, distList, k, depth + 1);

    double r = (double) abs(node->data[cur_dim] - target[cur_dim]);
    double R = distList.back().dist;

    if (distList.size() < k || r <= R) {
        kNearestNeighbourRecursive(opposite_branch, target, distList, k, depth + 1);
    }
}

kDTreeNode* kDTree::insertkNN(kDTreeNode * node,const vector<int> &point, int&label, int depth){
    if(node == nullptr){
        return new kDTreeNode(point, label,nullptr,nullptr);
    }

    /* not yet at leaf*/
    int cur_dim, data_tree, data_insert;
    
    cur_dim = depth % this->k;
    data_tree = node->data[cur_dim];
    data_insert = point[cur_dim];
    
    if(data_tree > data_insert){
        /* move to the left*/
        return insertkNN(node->left,point,label,depth+1);
    }
    /*move to the right*/
    return insertkNN(node->right,point,label,depth+1);

}

void kDTree::mergeSortkNN(vector<vector<int>> &pointList, vector<int> &labels, int dim, int left, int right){
    if(left < right){
        int mid = left + (right - left) / 2;
        mergeSortkNN(pointList,labels,dim,left,mid);
        mergeSortkNN(pointList,labels,dim,mid+1,right);
        mergekNN(pointList,labels,dim,left,mid,right);
    }
}

void kDTree::mergekNN(vector<vector<int>> &points, vector<int> &labels, int dim, int low, int mid, int high){
    //mid + 1 because mid is included in the left vector
    vector<vector<int>> left(points.begin() + low, points.begin() + mid + 1);
    vector<vector<int>> right(points.begin() + mid + 1, points.begin() + high + 1);
    vector<int> left_labels(labels.begin() + low, labels.begin() + mid + 1);
    vector<int> right_labels(labels.begin() + mid + 1, labels.begin() + high + 1);

    int i = 0, j = 0, main_vec = low;
    int left_size = (int)left.size();
    int right_size = (int)right.size();

    while(i < left_size && j < right_size){
        if(left[i][dim] < right[j][dim]){
            points[main_vec] = left[i];
            labels[main_vec++] = left_labels[i++];
        }
        else{
            points[main_vec] = right[j];
            labels[main_vec++] = right_labels[j++];
        }
    }

    //Handle the remaining elements
    while(i < left_size){
        points[main_vec] = left[i];
        labels[main_vec++] = left_labels[i++];
    }
    while (j < right_size){
        points[main_vec] = right[j];
        labels[main_vec++] = right_labels[j++];
    }

}

kDTreeNode* kDTree::buildTreekNNRecursive(kDTreeNode*node,vector<vector<int>> &pointList, vector<int> &labels, int depth){
    if(pointList.empty()){
        return nullptr;
    }
    int sz = (int)pointList.size();
    /*tested with odd and even, obeys policy*/
    int dim = depth % this->k;
    mergeSortkNN(pointList,labels,dim,0,sz-1);

    int median = (sz - 1) / 2;

    /* Median is not included since it is added as the new root (for the recursion)*/
    /* Add in preorder style: root left right*/
    vector<vector<int>> left(pointList.begin(),pointList.begin() + median);
    vector<vector<int>> right(pointList.begin() + median + 1,pointList.end());
    vector<int> left_labels(labels.begin(),labels.begin() + median);
    vector<int> right_labels(labels.begin() + median + 1,labels.end());

    kDTreeNode * newNode = insertkNN(node,pointList[median],labels[median],depth);
    newNode->left = buildTreekNNRecursive(newNode,left,left_labels,depth+1);
    newNode->right = buildTreekNNRecursive(newNode,right,right_labels,depth+1);
    return newNode;
}

void kDTree::buildTreekNN(const vector<vector<int>> &pointList,const vector<int> &labels){
    vector<vector<int>> pointsCopy = pointList;
    vector<int> labelsCopy = labels;
    root = buildTreekNNRecursive(root,pointsCopy,labelsCopy,0);
}


/*
 * *************************************
 *          KNN implementation
 * *************************************
*/
kNN::kNN(int k){
    this->k = k;
    this->X_train_tree = nullptr;
}

kNN::~kNN(){
    X_train_tree->~kDTree();
}

void kNN::fit(Dataset &X_train, Dataset &y_train){
    int rows, cols;
    X_train.getShape(rows,cols);
    
    /*Initialization*/
    vector<vector<int>> pointList(rows,vector<int>(cols,0));
    vector<int> labels(rows,0);
    list<list<int>>::iterator it = X_train.data.begin();
    list<list<int>>::iterator y_it = y_train.data.begin();
    list<int>::iterator it2;
    list<int>::iterator y_it2;
    /* fit */
    for(int i = 0; i < rows; i++, it++, y_it++){
        it2 = it->begin();
        y_it2 = y_it->begin();
        for(int j = 0; j < cols; j++, it2++){
            pointList[i][j] = *it2;
        }
        labels[i] = *y_it2;
    }
    X_train_tree = new kDTree(cols);
    X_train_tree->buildTreekNN(pointList,labels);
#ifdef FITDBG
    cout << "nodeCount: " << X_train_tree->nodeCount() << endl;
#endif
    label_name = y_train.columnName[0];
}

Dataset kNN::predict(Dataset &X_test){
    int rows, cols;
    X_test.getShape(rows,cols);
//#define PREDICTDBG
    Dataset y_pred;
    y_pred.columnName.push_back(label_name);

    vector<int> point(cols,0);
    list<list<int>>::iterator it = X_test.data.begin();
    list<int>::iterator it2;
    int count = 0;
    for(int i = 0; i < rows; i++, it++){
        it2 = it->begin();
#ifdef PREDICTDBG
        cout << "Row: " << i << endl;
#endif
        for(int j = 0; j < cols; j++, it2++){
            point[j] = *it2;
        }
        kDTreeNode *best = nullptr;
        vector<kDTreeNode*> bestList;
        int label[10] = {0};

        X_train_tree->kNearestNeighbour(point,k,bestList);
#ifdef PREDICTDBG
        cout << "Predict successful\n";
#endif
        count++;
        for(int j = 0; j < k; j++){
            label[bestList[j]->label]++;
        }
        int max = 0, pred = -1;
        for(int j = 0; j < 10; j++){
            if(label[j] > max){
                max = label[j];
                pred = j;
            }
        }
        list<int> pred_list;
        pred_list.push_back(pred);
        y_pred.data.push_back(pred_list);
    }
#ifdef PREDICTDBG
    cout << "Count - row: " << count << " " << rows << endl;
#endif
    return y_pred;
}

double kNN::score(const Dataset &y_test, const Dataset &y_pred){
    int rows, cols;
    y_test.getShape(rows,cols);
    list<list<int>>::const_iterator it = y_test.data.begin();
    list<list<int>>::const_iterator it2 = y_pred.data.begin();
    list<int>::const_iterator it3;
    list<int>::const_iterator it4;

    int correct = 0;
    for(int i = 0; i < rows; i++, it++, it2++){
        it3 = it->begin();
        it4 = it2->begin();
        if(*it3 == *it4){
            correct++;
        }
    }
    return (double)correct / rows;
}


