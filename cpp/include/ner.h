#ifndef NER_H
#define NER_H

#define NUM_LABEL 3
#define LABEL_LIST "NR", "NS", "NT"
#include <iostream>
#include <numeric>
#include <fstream>
#include <map>
#include <algorithm>
#include "onnxruntime_cxx_api.h"
#include "tokenizer.h"

struct Entity {
    std::string word;
    std::string label;
};

class NEROnnx
{
public:
    NEROnnx(const char *model_path, int nNumThread);
    ~NEROnnx();
    void inference(std::string text, int max_seq_length);
    void init_tokenizer(std::string model_path);
    void load_model(const std::string &model_path, int nNumThread);
    std::vector<Entity> bioToEntities(const std::vector<std::string>& bioTags, const std::vector<std::string>& words);
    // bool contains_unicode(std::string input);
    // std::string remove_unicode(const std::string &input);
    // void replaceUnicode(std::string &str, const std::string &from, const std::string &to);

private:

    Ort::Session *m_session;
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "TestNER");
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // tokenizer
    BertTokenizer tokenizer;

    std::vector<std::string> labels;
    std::map<std::string, int64_t> label2id;
    std::map<int64_t, std::string> id2label;

    std::vector<const char *> input_names{"input_ids", "attention_mask", "token_type_ids"};
    std::vector<const char *> output_names{"logits"};
};

#endif