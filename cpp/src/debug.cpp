#include <iostream>
#include "ner.h"
using namespace std;


template<typename T>
void printVector(vector<T> &v)
{

    for (typename vector<T>::iterator it = v.begin(); it != v.end(); it++)
    {
        cout << *it << " ";
    }
    cout << endl;
}


int main()
{
  NEROnnx ner("../onnx_output",1);
  // string text = "啊嗯对哦，OK有蓝色，有蓝的那个哦，有logo是吧？Logo对色花式咖啡嗯嗯行对嗯。";
  string text = "在哈佛大学发表重要演讲在纽约时出席大型晚宴并演讲";
  ner.inference(text, 256);
  return 0;

}