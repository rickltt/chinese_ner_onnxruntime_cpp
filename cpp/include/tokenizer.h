#include <iostream>
#include <map>
#include <vector>
#include <glog/logging.h>
#include <fstream>
#include <string>
#include <cctype>
#include <regex>
#include "unicode.h"
#include "uninorms.h"
#include <codecvt>
#include <algorithm>
#include <boost/regex/pending/unicode_iterator.hpp>
#include <boost/spirit/include/qi.hpp>
#include <cstdint>
std::vector<std::string> whitespace_tokenize(std::string text);

std::map<std::string, int> read_vocab(const char *filename);

class BasicTokenizer
{
public:
    bool do_lower_case_;
    std::vector<std::string> never_split_;

    BasicTokenizer(bool do_lower_case = false,
                   std::vector<std::string> never_split = {"[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"})
    {
        do_lower_case_ = do_lower_case;
        never_split_ = never_split;
    }

    std::string _clean_text(std::string text);

    std::vector<std::string> _run_split_on_punc(std::string text);

    std::string _run_strip_accents(std::string text);

    std::string _tokenize_chinese_chars(std::string text);

    std::string utf8chr(int cp);

    bool _is_chinese_char(int cp);

    std::vector<std::string> tokenize(std::string text);

    void truncate_sequences(
            std::vector<std::string> &textA, std::vector<std::string> &textB, const char *truncation_strategy, int max_seq_length);
};

class WordpieceTokenizer
{
public:
    std::map<std::string, int> vocab_;
    std::string unk_token_;
    int max_input_chars_per_word_;

    WordpieceTokenizer() {};

    WordpieceTokenizer(std::map<std::string, int> vocab, std::string unk_token = "[UNK]", int max_input_chars_per_word = 100)
    {
        vocab_ = vocab;
        unk_token_ = unk_token;
        max_input_chars_per_word_ = max_input_chars_per_word;
    }

    void add_vocab(std::map<std::string, int> vocab);

    std::vector<std::string> tokenize(std::string text);
};


class BertTokenizer
{
public:
    std::map<std::string, int> vocab;
    std::map<int, std::string> ids_to_tokens;
    bool do_lower_case_;
    bool do_basic_tokenize_;
    int maxlen_;
    BasicTokenizer basic_tokenizer;
    WordpieceTokenizer wordpiece_tokenizer;

    BertTokenizer() {};

    BertTokenizer(const char *vocab_file, bool do_lower_case = false, int max_len = 512, bool do_basic_tokenize = true,
                  std::vector<std::string> never_split = {"[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"})
    {
        vocab = read_vocab(vocab_file);
        for (std::map<std::string, int>::iterator i = vocab.begin(); i != vocab.end(); ++i)
            ids_to_tokens[i->second] = i->first;
        do_basic_tokenize_ = do_basic_tokenize;
        do_lower_case_ = do_lower_case;
        wordpiece_tokenizer.add_vocab(vocab);
        maxlen_ = max_len;
    }

    void add_vocab(const char *vocab_file);

    std::vector<std::string> tokenize(std::string text);

    std::vector<int64_t> convert_tokens_to_ids(std::vector<std::string> tokens);

    void encode(std::string text_, std::vector<int64_t> &input_ids, std::vector<int64_t> &input_mask, std::vector<int64_t> &segment_ids,
           int max_seq_length = 512);
};