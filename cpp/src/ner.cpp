#include "ner.h"

template <class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

std::vector<Entity> NEROnnx::bioToEntities(const std::vector<std::string>& bioTags, const std::vector<std::string>& words) {
    std::vector<Entity> entities;
    Entity currentEntity;
    
    for (size_t i = 0; i < bioTags.size(); ++i) {
        const std::string& tag = bioTags[i];

        if (tag.substr(0, 2) == "B-") {  // 开始实体
            if (!currentEntity.word.empty()) {
                entities.push_back(currentEntity);
            }
            currentEntity.word = words[i];
            currentEntity.label = tag.substr(2);
        } else if (tag.substr(0, 2) == "I-") {  // 中间实体
            if (!currentEntity.word.empty()) {
                currentEntity.word += " " + words[i];
            }
        } else {  // O 或不属于当前实体
            if (!currentEntity.word.empty()) {
                entities.push_back(currentEntity);
                currentEntity = Entity();  // 重置当前实体
            }
        }
    }
    
    // 添加最后的实体
    if (!currentEntity.word.empty()) {
        entities.push_back(currentEntity);
    }

    return entities;
}

NEROnnx::NEROnnx(const char *model_path, int nNumThread)
{
    labels.push_back("O");
    const char* label_list[NUM_LABEL] = { LABEL_LIST };
    for (int i = 0; i < NUM_LABEL; ++i){
        std::string label_name = label_list[i];
        labels.push_back("B-"+label_name);
        labels.push_back("I-"+label_name);
    }
    for (int i = 0; i < labels.size(); ++i) {
        label2id.insert(std::pair<std::string, int64_t>(labels[i], i));
        id2label.insert(std::pair<int64_t, std::string>(i,labels[i]));
    }
    init_tokenizer(model_path);
    load_model(model_path, nNumThread);
}

NEROnnx::~NEROnnx()
{

    if (m_session)
    {
        delete m_session;
        m_session = nullptr;
    }
}

void NEROnnx::load_model(const std::string &model_path, int nNumThread)
{
    sessionOptions.SetIntraOpNumThreads(nNumThread);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sessionOptions.DisableMemPattern();
    sessionOptions.DisableCpuMemArena();

    std::string strModelPath = model_path + "/" + "model_quant.onnx";

    m_session = new Ort::Session(env, strModelPath.c_str(), sessionOptions);
    LOG(INFO) << "Load Model";
}


// bool NEROnnx::contains_unicode(std::string input)
// {
//     for (char ch : input)
//     {
//         // ASCII characters are in the range 0x00 to 0x7F
//         if (static_cast<unsigned char>(ch) > 0x7F)
//         {
//             return true;
//         }
//     }
//     return false;
// }

// std::string NEROnnx::remove_unicode(const std::string &input)
// {
//     std::string output;
//     for (char ch : input)
//     {
//         // ASCII characters are in the range 0x00 to 0x7F
//         if (static_cast<unsigned char>(ch) <= 0x7F)
//         {
//             output += ch;
//         }
//     }
//     return output;
// }

// void NEROnnx::replaceUnicode(std::string &str, const std::string &from, const std::string &to)
// {
//     size_t start_pos = 0;
//     while ((start_pos = str.find(from, start_pos)) != std::string::npos)
//     {
//         str.replace(start_pos, from.length(), to);
//         // Advance start_pos to avoid replacing the same substring again
//         start_pos += to.length();
//     }
// }

void NEROnnx::init_tokenizer(std::string model_path)
{

    std::string vocab_path = model_path + "/vocab.txt";
    tokenizer.add_vocab(vocab_path.c_str());
    LOG(INFO) << "Load Vocab";
}

void NEROnnx::inference(std::string text, int max_seq_length)
{
    std::vector<std::string> tokens = tokenizer.tokenize(text);
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<int64_t> token_type_ids;
    tokenizer.encode(text, input_ids, attention_mask, token_type_ids, max_seq_length);

    std::array<int64_t, 2> input_ids_shape{1, (int64_t)input_ids.size()};
    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, input_ids.data(), input_ids.size(),
                                                                    input_ids_shape.data(), input_ids_shape.size());

    std::array<int64_t, 2> attention_mask_shape{1, (int64_t)attention_mask.size()};
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, attention_mask.data(), attention_mask.size(),
                                                                         attention_mask_shape.data(), attention_mask_shape.size());

    std::array<int64_t, 2> token_type_ids_shape{1, (int64_t)token_type_ids.size()};
    Ort::Value token_type_ids_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, token_type_ids.data(), token_type_ids.size(),
                                                                         token_type_ids_shape.data(), token_type_ids_shape.size());
    std::vector<Ort::Value> input_onnx;
    input_onnx.emplace_back(std::move(input_ids_tensor));
    input_onnx.emplace_back(std::move(attention_mask_tensor));
    input_onnx.emplace_back(std::move(token_type_ids_tensor));


    auto outputTensor = m_session->Run(Ort::RunOptions(), input_names.data(), input_onnx.data(), input_names.size(), output_names.data(), output_names.size());
    std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
    float *logits = outputTensor[0].GetTensorMutableData<float>();

    int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
    std::vector<int64_t> predictions;
    int num_labels = 2 * NUM_LABEL + 1;
    for (size_t i = 0; i < outputCount; i += num_labels)
    {
      int64_t index = argmax(logits + i, logits + i + num_labels - 1);
      predictions.push_back(index);
    }
    predictions.assign(predictions.begin()+1, predictions.begin() + std::count(attention_mask.begin(), attention_mask.end(), 1) -1 );
    
    std::vector<std::string> bioTags;
    for(auto &pred: predictions){
        bioTags.push_back(id2label[pred]);
        LOG(INFO) << pred << " " << id2label[pred];
    }
    // 转换 BIO 到实体
    std::vector<Entity> entities = bioToEntities(bioTags, tokens);
    // 输出结果
    for (const auto& entity : entities) {
        std::cout << "实体: " << entity.word << ", 标签: " << entity.label << std::endl;
    }

}
