import pickle

def main():
    # give each category a ID
    category_name_un = ['FW', '-LRB-', '-RRB-', 'LS']  # 1不明白
    category_name_vb = ['VB', 'VBD', 'VBP', 'VBG', 'VBN', 'VBZ']  # 2动词
    category_name_nn = ['NN', 'NNS', 'NNP']  # 3名词
    category_name_jj = ['JJ', 'JJR', 'JJS']  # 4形容词
    category_name_rb = ['RB', 'RBS', 'RBR', 'WRB', 'EX']  # 5副词
    category_name_cc = ['CC']  # 6连词
    category_name_pr = ['PRP', 'PRP$', 'WP', 'POS', 'WP$']  # 7代词
    category_name_in = ['IN', 'TO']  # 8介词
    category_name_dt = ['DT', 'WDT', 'PDT']  # 9冠词
    category_name_rp = ['RP', 'MD']  # 10助词
    category_name_cd = ['CD']  # 11数字
    category_name_sy = ['SYM', ':', '``', '#', '$']  # 12符号
    category_name_uh = ['UH']  # 13叹词

    PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
    UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
    START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
    STOP_DECODING = '[END]'  #

    msvd_pos = {}
    _pos_to_id = {}
    _id_to_pos = {}
    _pcount = 0

    for p in [PAD_TOKEN, START_DECODING, STOP_DECODING, UNKNOWN_TOKEN]:
        _pos_to_id[p] = _pcount
        _id_to_pos[_pcount] = p
        _pcount += 1

    cap_pos_path = "/data1/tuyunbin/PR/data/msrvtt16/CAP_with_POS.pkl"
    with open(cap_pos_path, 'rb') as f:
        caption = pickle.load(f, encoding='iso-8859-1')
        for kk, vv in caption.items():
            for i in vv:
                pos = i['pos']
                for p in pos:
                    if p in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                        raise Exception(
                            '[UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                    if p in category_name_vb:
                        _pos_to_id[p] = 4
                        _id_to_pos[4] = p
                    elif p in category_name_nn:
                        _pos_to_id[p] = 5
                        _id_to_pos[5] = p
                    elif p in category_name_jj:
                        _pos_to_id[p] = 6
                        _id_to_pos[6] = p
                    elif p in category_name_rb:
                        _pos_to_id[p] = 7
                        _id_to_pos[7] = p

                    elif p in category_name_cc:
                        _pos_to_id[p] = 8
                        _id_to_pos[8] = p

                    elif p in category_name_pr:
                        _pos_to_id[p] = 9
                        _id_to_pos[9] = p

                    elif p in category_name_in:
                        _pos_to_id[p] = 10
                        _id_to_pos[10] = p

                    elif p in category_name_dt:
                        _pos_to_id[p] = 11
                        _id_to_pos[11] = p
                    elif p in category_name_rp:
                        _pos_to_id[p] = 12
                        _id_to_pos[12] = p
                    elif p in category_name_cd:
                        _pos_to_id[p] = 13
                        _id_to_pos[13] = p
                    elif p in category_name_sy:
                        _pos_to_id[p] = 14
                        _id_to_pos[14] = p
                    elif p in category_name_uh:
                        _pos_to_id[p] = 15
                        _id_to_pos[15] = p
                    else:
                        _pos_to_id[p] = 3
                        _id_to_pos[3] = p
    msvd_pos["pos_to_id"] = _pos_to_id
    msvd_pos["id_to_pos"] = _id_to_pos
    with open("/data1/tuyunbin/PR/data/msrvtt16/msr_pos.pkl", 'wb') as fo:  # 将数据写入pkl文件
        pickle.dump(msvd_pos, fo)

if __name__ == "__main__":
    main()
