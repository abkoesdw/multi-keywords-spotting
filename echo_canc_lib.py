import numpy as np


def create_time_series(in_, out_, lag_):
    data_in, data_out = [], []
    _, col_ = np.shape(in_)
    in_ = np.concatenate((np.zeros((lag_, col_)), in_), axis=0)
    out_ = np.concatenate((np.zeros((lag_, col_)), out_), axis=0)
    row_, _ = np.shape(in_)
    for i in range(row_ - lag_):
        a = in_[i:i + lag_ + 1, :]
        if i == 0:
            data_in = a[::-1].reshape((col_ * lag_ + col_, 1))
            data_out = out_[i + lag_, :].reshape((col_, 1))
            # print dataY
        else:
            data_in = np.append(data_in, a[::-1].reshape((col_ * lag_ + col_, 1)), axis=1)
            data_out = np.append(data_out, out_[i + lag_, :].reshape((col_, 1)), axis=1)

    return data_in.T, data_out.T


def get_energy(dataset_, sqrt):
    rec_real = dataset_['rec_real']
    rec_im = dataset_['rec_im']
    src_real = dataset_['src_real']
    src_im = dataset_['src_im']
    rec_temp = rec_real ** 2 + rec_im ** 2
    src_temp = src_real ** 2 + src_im ** 2
    rec_ = rec_temp
    src_ = src_temp
    if sqrt == 1:
        rec_ = np.sqrt(rec_temp)
        src_ = np.sqrt(src_temp)

    input_temp = src_
    output_temp = rec_
    return input_temp, output_temp


def get_complex_num(y_temp, output_energy, rec_real, rec_im):
    row_out, col_out = output_energy.shape
    y_ = np.zeros((row_out, col_out * 2))
    y_sub = np.zeros((row_out, col_out * 2))
    y_orig = np.zeros((row_out, col_out * 2))
    for data_point in range(row_out):
        for en_band in range(col_out):
            if output_energy[data_point, en_band] == 0:
                y_temp_real = 0
                y_temp_imag = 0
                real_ = 0
                imag_ = 0
            else:
                orig_real = (rec_real[data_point, en_band] ** 2)
                pred_real = (rec_real[data_point, en_band] ** 2) * \
                            ((y_temp[data_point, en_band] / output_energy[data_point, en_band]) ** 2)
                real_sub = orig_real - pred_real

                orig_im = (rec_im[data_point, en_band] ** 2)
                pred_im = (rec_im[data_point, en_band] ** 2) * \
                          ((y_temp[data_point, en_band] / output_energy[data_point, en_band]) ** 2)
                imag_sub = orig_im - pred_im

                if real_sub < 0:
                    real_sub = 0
                if imag_sub < 0:
                    imag_sub = 0

                real_sub = np.sqrt(real_sub)
                imag_sub = np.sqrt(imag_sub)

                if rec_real[data_point, en_band] > 0:
                    real_sub = real_sub
                if rec_im[data_point, en_band] > 0:
                    imag_sub = imag_sub
                if rec_real[data_point, en_band] < 0:
                    real_sub = - real_sub
                if rec_im[data_point, en_band] < 0:
                    imag_sub = - imag_sub

                y_temp_real = pred_real
                y_temp_imag = pred_im
                y_orig_real = orig_real
                y_orig_imag = orig_im

                y_temp_real = np.sqrt(y_temp_real)
                y_temp_imag = np.sqrt(y_temp_imag)
                y_orig_real = np.sqrt(y_orig_real)
                y_orig_imag = np.sqrt(y_orig_imag)

                if rec_real[data_point, en_band] > 0:
                    y_temp_real = y_temp_real
                    y_orig_real = y_orig_real
                if rec_im[data_point, en_band] > 0:
                    y_temp_imag = y_temp_imag
                    y_orig_imag = y_orig_imag
                if rec_real[data_point, en_band] < 0:
                    y_temp_real = - y_temp_real
                    y_orig_real = - y_orig_real
                if rec_im[data_point, en_band] < 0:
                    y_temp_imag = - y_temp_imag
                    y_orig_imag = - y_orig_imag

            y_[data_point, en_band] = y_temp_real
            y_[data_point, en_band+256] = y_temp_imag
            y_sub[data_point, en_band] = real_sub
            y_sub[data_point, en_band+256] = imag_sub
            y_orig[data_point, en_band] = y_orig_real
            y_orig[data_point, en_band + 256] = y_orig_imag

    return y_, y_sub, y_orig


def get_complex_num_demo(y_temp, output_energy, rec_real, rec_im):
    row_out, col_out = output_energy.shape
    y_sub = np.zeros((row_out, col_out * 2))
    for data_point in range(row_out):
        for en_band in range(col_out):
            if output_energy[data_point, en_band] != 0:
                orig_real = (rec_real[data_point, en_band] ** 2)
                pred_real = (rec_real[data_point, en_band] ** 2) * \
                            ((y_temp[data_point, en_band] / output_energy[data_point, en_band]) ** 2)
                real_sub = orig_real - pred_real

                orig_im = (rec_im[data_point, en_band] ** 2)
                pred_im = (rec_im[data_point, en_band] ** 2) * \
                            ((y_temp[data_point, en_band] / output_energy[data_point, en_band]) ** 2)
                imag_sub = orig_im - pred_im

                if real_sub < 0:
                    real_sub = 0
                if imag_sub < 0:
                    imag_sub = 0

                real_sub = np.sqrt(real_sub)
                imag_sub = np.sqrt(imag_sub)

                if rec_real[data_point, en_band] > 0:
                    real_sub = real_sub
                if rec_im[data_point, en_band] > 0:
                    imag_sub = imag_sub
                if rec_real[data_point, en_band] < 0:
                    real_sub = - real_sub
                if rec_im[data_point, en_band] < 0:
                    imag_sub = - imag_sub

            y_sub[data_point, en_band] = real_sub
            y_sub[data_point, en_band+256] = imag_sub

    return y_sub


def get_labels(line_):
    keyword = {}
    start_frame = {}
    end_frame = {}
    kw_lst = line_.split()
    len_word = len(kw_lst)
    filename_ = kw_lst[0][7:]
    i = 1
    j = 0
    while i < len_word:
        if kw_lst[i] == 'system' or kw_lst[i] == 'systems':
            keyword[str(j)] = kw_lst[i]
            start_frame[str(j)] = int(kw_lst[i + 1])/100
            end_frame[str(j)] = int(kw_lst[i + 2])/100
            i += 3
            j += 1
        else:
            i += 3

    return filename_, keyword, start_frame, end_frame


def get_feature(mfcc_feat, left_context, right_context):
    num_frames, num_cep = mfcc_feat.shape
    feature = np.array([])
    for j in range(num_frames):
        current_frame = mfcc_feat[j, :].reshape(1, num_cep)
        if j < left_context and j + right_context <= num_frames:
            frame_temp = np.tile(mfcc_feat[0, :], (left_context - j, 1))
            if not mfcc_feat[0:j, :].any():
                left_frame = frame_temp
            else:
                left_frame = np.concatenate((frame_temp, mfcc_feat[0:j, :].reshape(len(mfcc_feat[0:j, :]), num_cep)),
                                            axis=0)

            right_frame = mfcc_feat[j + 1:j + 1 + right_context, :].reshape(right_context, num_cep)

        elif j >= left_context and j + right_context < num_frames:
            left_frame = mfcc_feat[j - left_context:j, :].reshape(left_context, num_cep)
            right_frame = mfcc_feat[j + 1:j + 1 + right_context, :].reshape(right_context, num_cep)

        else:
            left_frame = mfcc_feat[j - left_context:j, :].reshape(left_context, num_cep)
            frame_temp = np.tile(mfcc_feat[-1, :], (j + 1 + right_context - num_frames, 1))

            right_frame = np.concatenate((mfcc_feat[j + 1:, :].reshape(len(mfcc_feat[j + 1:, :]), num_cep),
                                          frame_temp)).reshape(right_context, num_cep)

        feature_temp = np.concatenate((left_frame, current_frame, right_frame), axis=0)
        feature_temp = feature_temp.reshape((left_context+right_context+1)*num_cep, 1)

        if j == 0:
            feature = feature_temp
        else:
            feature = np.concatenate((feature, feature_temp), axis=1)

    return feature


def get_feature_new(mfcc_feat, left_context, right_context):
    num_frames, num_cep = mfcc_feat.shape
    feature = np.empty((num_cep*(left_context+right_context+1), 0))
    for j in range(4, num_frames-right_context):
        current_frame = mfcc_feat[j, :].reshape(num_cep, 1)
        left_frame = mfcc_feat[j-3:j, :].reshape(num_cep*left_context, 1)
        right_frame = mfcc_feat[j+1:j+1+right_context, :].reshape(num_cep*right_context, 1)
        total_frame = np.concatenate((current_frame, left_frame, right_frame), axis=0)
        feature = np.concatenate((feature, total_frame), axis=1)

    return feature.T

def get_feature_multi(mfcc_feat, left_context, right_context):
    num_frames, num_cep = mfcc_feat.shape
    feature = np.empty((num_cep*(left_context+right_context+1), 0))
    for j in range(4, num_frames-right_context):
        current_frame = mfcc_feat[j, :].reshape(num_cep, 1)
        left_frame = mfcc_feat[j-3:j, :].reshape(num_cep*left_context, 1)
        right_frame = mfcc_feat[j+1:j+1+right_context, :].reshape(num_cep*right_context, 1)
        total_frame = np.concatenate((current_frame, left_frame, right_frame), axis=0)
        feature = np.concatenate((feature, total_frame), axis=1)

    return feature.T


def get_label_new(start_frame, end_frame, total_frame, left_context, right_context):

    label = np.zeros((int(total_frame), 1))
    for i in range(len(start_frame)):
        label[int(start_frame[i]):int(end_frame[i])+1] = 1
    label = label[4:-right_context]

    return label


def get_label_multi(line_split, total_frame, right_context):
    labels = ['econom', 'financ', 'movie', 'music', 'news', 'resume', 'scien',
              'sport', 'stop', 'world', 'us']
    num_frames = total_frame
    target_temp = np.zeros((1, num_frames))
    start_frame = 0
    end_frame = 0
    flag_music = 0
    for k in range(1, len(line_split), 3):
        label_val = 1
        for label in labels:
            if label in line_split[k] and label is not "us":
                start_frame = line_split[k + 1]
                end_frame = line_split[k + 2]
                target_temp[0, int(start_frame):int(end_frame) + 1] = int(label_val)

            elif label == line_split[k] and label is "us":
                start_frame = line_split[k + 1]
                end_frame = line_split[k + 2]
                target_temp[0, int(start_frame):int(end_frame) + 1] = int(label_val)

            label_val += 1

    target = target_temp[0, 4:-right_context]

    return np.int16(target)


def get_label_multi_prev(keyword, start_frame, end_frame, right_context, prev_label):
    labels = ['econom', 'financ', 'movie', 'music', 'news', 'resume', 'scien',
              'sport', 'stop', 'world', 'us']
    m = 1
    label_val = 0
    for label_item in labels:
        if label_item in keyword:
            label_val = m
        m += 1
    label = prev_label
    label[int(start_frame):int(end_frame)+1] = int(label_val)
    label_prev = label
    label_final = label[4:-right_context]

    return np.int16(label_final), np.int16(label_prev)


def get_label(feature, keyword, start_frame, end_frame):
    num_feat, num_frames = feature.shape

    label = np.zeros((1, num_frames))
    for l in range(len(keyword)):
        start_fr = int(start_frame[str(l)] / 1)
        end_fr = int(end_frame[str(l)] / 1)
        # len_fr = end_fr - start_fr
        # idx_st = start_fr - (len_fr/2)
        # idx_end = end_fr + (len_fr / 2)
        # if keyword[str(l)] == "ok" or keyword[str(l)] == "okay":
        #     label[0, start_fr:end_fr + 1] = 1
        if keyword[str(l)] == "system" or keyword[str(l)] == "systems":
            label[0, start_fr:end_fr + 1] = 1

    return label


def get_avg_lbl(keyword, start_frame, end_frame):
    sum_ok = 0
    sum_system = 0
    # print keyword
    if keyword:
        # print keyword
        for l in range(len(keyword)):
            start_fr = start_frame[str(l)]
            end_fr = end_frame[str(l)]
            # if keyword[str(l)] == "ok" or keyword[str(l)] == "okay":
            #     sum_ok = (end_fr - start_fr)
            if keyword[str(l)] == "system" or keyword[str(l)] == "systems":
                sum_system = (end_fr - start_fr)

    return sum_ok, sum_system


def align_wav(data1, data2):
    # mod_data1 = len(data1) % 512
    # mod_data2 = len(data2) % 512
    # print mod_data1, mod_data2
    # if mod_data1 != 0:
    #     data1 = np.append(data1, np.zeros((512 - mod_data1)))
    #
    # if mod_data2 != 0:
    #     data2 = np.append(data2, np.zeros((512 - mod_data2)))

    data2_flipped = data2[::-1]
    data1_fft = np.fft.rfft(data1)
    data2_fft = np.fft.rfft(data2_flipped)

    R1 = np.fft.irfft(data1_fft * data2_fft)
    R1 = R1.real

    # find the lag
    idx_max = np.argmax(R1)
    time_lag = np.arange(-len(data1) + 1, len(data1))
    time_lag = np.reshape(time_lag, (len(time_lag), 1))
    lag_temp = time_lag[idx_max]
    lag = lag_temp[0]

    data2_shifted = data2[-lag + 1:]
    return data2_shifted, lag

def align_wav_live(data1, data2, prev_source):
    # mod_data1 = len(data1) % 512
    # mod_data2 = len(data2) % 512
    # print mod_data1, mod_data2
    # if mod_data1 != 0:
    #     data1 = np.append(data1, np.zeros((512 - mod_data1)))
    #
    # if mod_data2 != 0:
    #     data2 = np.append(data2, np.zeros((512 - mod_data2)))

    data2_flipped = data2[::-1]
    data1_fft = np.fft.rfft(data1)
    data2_fft = np.fft.rfft(data2_flipped)

    R1 = np.fft.irfft(data1_fft * data2_fft)
    R1 = R1.real

    # find the lag
    idx_max = np.argmax(R1)
    time_lag = np.arange(-len(data1) + 1, len(data1))
    time_lag = np.reshape(time_lag, (len(time_lag), 1))
    lag_temp = time_lag[idx_max]
    lag = lag_temp[0]

    # data2_shifted = data2[-lag + 1:]
    # print(lag)
    data1_shifted = np.append(prev_source[lag:], data1[:lag])
    # print(data1_shifted.shape, data2.shape)
    return data1_shifted, lag


def get_fft(w1, w2):
    mod_w1 = len(w1) % 512
    # print mod_w1, w1.shape
    mod_w2 = len(w2) % 512
    if mod_w1 != 0:
        w1 = np.concatenate((w1, np.zeros((512 - mod_w1, 1))))

    # if mod_w2 != 0:
    len_diff = len(w1) - len(w2)
    # print len_diff
    w2 = np.concatenate((w2, np.zeros((len_diff, 1))))
    # print mod_w2, len(w1), len(w2)
    num_samples = 512
    step_size = 256
    i = 0
    temp1 = np.array([])
    temp2 = np.array([])
    while i < len(w1):
        temp_w1 = w1[i:i + num_samples]
        temp_w2 = w2[i:i + num_samples]
        # print k, temp_w1.shape, w1.shape, w2.shape
        M = np.hamming(num_samples).reshape(num_samples, 1)
        # print len(temp_w1), len(temp_w2),len(M)
        if len(temp_w1) != len(M):
            M = M[0:256]

        temp_w1 = np.multiply(temp_w1, M)
        temp_w2 = np.multiply(temp_w2, M)
        # print temp_w1.shape
        if len(temp_w1) != 512:
            # print np.zeros((512-len(temp_w1), 1))
            temp_w1 = np.concatenate((temp_w1, np.zeros(((512 - len(temp_w1)), 1))))
            temp_w2 = np.concatenate((temp_w2, np.zeros(((512 - len(temp_w2)), 1))))

        temp_w1_fft = np.fft.rfft(temp_w1, axis=0)  # /len(temp_w1)
        temp_w1_fft = temp_w1_fft[0:256, :]
        temp_w2_fft = np.fft.rfft(temp_w2, axis=0)  # /len(temp_w2)
        temp_w2_fft = temp_w2_fft[0:256, :]
        if i == 0:
            temp1 = np.append(temp1, temp_w1_fft)
            w1_fft = np.reshape(temp1, (256, 1))
            temp2 = np.append(temp2, temp_w2_fft)
            w2_fft = np.reshape(temp2, (256, 1))
        else:
            # print w1_fft.shape, temp_w1_fft.shape
            w1_fft = np.append(w1_fft, temp_w1_fft, axis=1)
            w2_fft = np.append(w2_fft, temp_w2_fft, axis=1)
        i += step_size

    w1_fft_real = np.transpose(w1_fft.real)
    w1_fft_imag = np.transpose(w1_fft.imag)
    w2_fft_real = np.transpose(w2_fft.real)
    w2_fft_imag = np.transpose(w2_fft.imag)
    return w1_fft_real, w1_fft_imag, w2_fft_real, w2_fft_imag


def compute_energy(src_real, src_im, rec_real, rec_im):
    targ1 = rec_real
    targ2 = rec_im
    targ = targ1 ** 2 + targ2 ** 2
    targ = np.sqrt(targ)

    feat1 = src_real
    feat2 = src_im
    feat = feat1 ** 2 + feat2 ** 2
    feat = np.sqrt(feat)

    return feat, targ


def invert_energy(input_, output_, output_est, output_sub):
    freq = np.fft.fftfreq(512, 1.000 / 16000)

    idx1 = np.where(freq == 4000)[0][0]
    idx2 = np.where(freq == -4000)[0][0]
    file_type = ['source', 'recorded', 'predicted', 'subtracted', 'subtracted_energy']

    w_fft = {}
    w1_fft_real = input_[:, 0:256]
    w1_fft_imag = input_[:, 256:512]
    w_fft['1', 'real'] = w1_fft_real
    w_fft['1', 'imag'] = w1_fft_imag

    w2_fft_real = output_[:, 0:256]
    w2_fft_imag = output_[:, 256:]
    w_fft['2', 'real'] = w2_fft_real
    w_fft['2', 'imag'] = w2_fft_imag

    w3_fft_real = output_est[:, 0:256]
    w3_fft_imag = output_est[:, 256:]
    w_fft['3', 'real'] = w3_fft_real
    w_fft['3', 'imag'] = w3_fft_imag

    w5_fft_real = output_sub[:, 0:256]
    w5_fft_imag = output_sub[:, 256:]
    w_fft['5', 'real'] = w5_fft_real
    w_fft['5', 'imag'] = w5_fft_imag

    gain = 1
    w4_fft_real = w2_fft_real - w3_fft_real * gain
    w4_fft_imag = w2_fft_imag - w3_fft_imag * gain
    w_fft['4', 'real'] = w4_fft_real
    w_fft['4', 'imag'] = w4_fft_imag

    w_temp = {}
    divisor = 1
    w = {}
    for k in range(1, 6):
        temp_1 = w_fft[str(k), 'real'] + 1j * w_fft[str(k), 'imag']
        temp1_flipped = np.fliplr(w_fft[str(k), 'real'])
        temp2_flipped = np.fliplr(w_fft[str(k), 'imag'])
        temp_2 = temp1_flipped + 1j * temp2_flipped
        temp = np.concatenate((temp_1, temp_2), axis=1)
        temp[:, idx1:idx2 + 1] = temp[:, idx1:idx2 + 1] / divisor
        w_temp[str(k)] = temp
        w[file_type[k - 1]] = w_temp[str(k)]

    wav_reconstructed = dict()
    for file__ in file_type:
        print("\n inverting", file__, "wav")
        wave_file = w[file__]
        temp_ = []

        for k in range(0, len(wave_file)):
            temp_wave = np.reshape(wave_file[k, :], (len(wave_file[k, :]), 1))
            temp_wave_ifft = np.fft.irfft(temp_wave, n=len(temp_wave), axis=0)

            if k == 0:
                temp_ = np.append(temp_wave_ifft, np.zeros((1, 256)))
            else:
                temp2 = np.append(np.zeros((1, k * 256)), temp_wave_ifft)
                temp_ += temp2
                temp_ = np.append(temp_, np.zeros((1, 256)))

        wav_reconstructed[file__] = np.array(temp_, dtype='int16')

    return wav_reconstructed


def invert_energy_demo(output_sub):
    freq = np.fft.fftfreq(512, 1.000 / 16000)

    idx1 = np.where(freq == 4000)[0][0]
    idx2 = np.where(freq == -4000)[0][0]

    w_real = output_sub[:, 0:256]
    w_imag = output_sub[:, 256:]

    divisor = 10000

    temp_1 = w_real + 1j * w_imag
    temp1_flipped = np.fliplr(w_real)
    temp2_flipped = np.fliplr(w_imag)
    temp_2 = temp1_flipped + 1j * temp2_flipped
    temp = np.concatenate((temp_1, temp_2), axis=1)
    temp[:, idx1:idx2 + 1] = temp[:, idx1:idx2 + 1] / divisor
    wave_file = temp
    temp_ = []
    for k in range(0, len(wave_file)):
        temp_wave = np.reshape(wave_file[k, :], (len(wave_file[k, :]), 1))
        temp_wave_ifft = np.fft.irfft(temp_wave, n=len(temp_wave), axis=0)

        if k == 0:
            temp_ = np.append(temp_wave_ifft, np.zeros((1, 256)))
        else:
            temp2 = np.append(np.zeros((1, k * 256)), temp_wave_ifft)
            temp_ += temp2
            temp_ = np.append(temp_, np.zeros((1, 256)))

    wav_reconstructed = np.array(temp_, dtype='int16')

    return wav_reconstructed


def compute_posterior(y_est, w_smooth, w_max):
    num_data, _ = y_est.shape
    j = 0
    y_smooth = np.zeros((num_data, 2))
    conf = np.zeros(num_data)
    while j < num_data:
        h_smooth = np.max([0, j - w_smooth + 1])
        den_post = j - h_smooth + 1
        sum_post = np.sum(y_est[h_smooth:j + 1], axis=0)
        y_smooth[j, :] = (1 / float(den_post)) * sum_post

        # confidence
        h_max = np.max([0, j - w_max + 1])
        max_p = np.max(y_smooth[h_max:j + 1, :], axis=0)
        prod_max = np.prod(max_p)

        conf[j] = prod_max
        j += 1

    return conf, y_smooth


def find_hotword(conf, win_size, frame_size, threshold):
    j = 0
    lw_frame_start = {}
    lw_frame_end = {}
    glob_frame = np.zeros(len(conf))
    k = 0
    while j < len(conf):
        mov_wind = conf[j:j+win_size]
        idx_winner = np.argmax(mov_wind)
        if mov_wind[idx_winner] >= threshold:
            if j + idx_winner - frame_size < 0:
                fr_start = j
                fr_end = j + idx_winner + frame_size + 1
            elif j + idx_winner + frame_size + 1 > len(conf):
                fr_end = len(conf) - 1
                fr_start = j + idx_winner - frame_size
            else:
                fr_start = j + idx_winner - frame_size
                fr_end = j + idx_winner + frame_size + 1
            lw_frame_start[str(k)] = fr_start
            lw_frame_end[str(k)] = fr_end
            k += 1
        j += win_size

    return lw_frame_start, lw_frame_end, glob_frame


def get_groundtruth_demo(line_):
    start_frame = {}
    end_frame = {}
    keyword = {}
    kw_lst = line_
    len_word = len(kw_lst)
    i = 0
    j = 0
    while i < len_word:
        if kw_lst[i] == 'system' or kw_lst[i] == 'systems':
            keyword[str(j)] = kw_lst[i]
            start_frame[str(j)] = int(kw_lst[i + 1])/100
            end_frame[str(j)] = int(kw_lst[i + 2])/100
            i += 3
            j += 1
        else:
            i += 3

    return start_frame, end_frame, keyword


def find_final_keyword(est_start, est_end, est_frame, act_start, act_end,  ov_lap_threshold):
    act_frame = est_frame
    num_est_kw = len(est_start)
    num_act_kw = len(act_start)
    est_kw = np.zeros(num_est_kw)
    act_kw = np.zeros(num_act_kw)
    if num_est_kw >= num_act_kw:
        est_kw = np.zeros(num_est_kw)
        act_kw = np.zeros(num_est_kw)
    elif num_est_kw < num_act_kw:
        est_kw = np.zeros(num_act_kw)
        act_kw = np.zeros(num_act_kw)

    j = 0
    for key1 in act_start:
        act_frame[act_start[key1]:act_end[key1]] = 1
        idx_act = np.where(act_frame == 1)[0]
        act_kw[j] = 1
        for key2 in est_start:
            est_frame[est_start[key2]:est_end[key2]] = 1
            idx_est = np.where(est_frame == 1)[0]
            overlapping = np.intersect1d(idx_est, idx_act)
            if len(overlapping)/len(act_frame) >= ov_lap_threshold:
                est_kw[j] = 1


def find_sequence(input_):
    j = 1
    seq = {}
    seq_temp = []
    prev_input = 0
    for i in range(len(input_)):
        if input_[i] == 1:
            seq_temp = np.append(seq_temp, i)
            if i == len(input_) - 1:

                seq[str(j)] = np.int16(seq_temp)
            prev_input = input_[i]

        if input_[i] == 0 and i != 0 and prev_input == 1:
            prev_input = input_[i]
            seq[str(j)] = np.int16(seq_temp)
            seq_temp = []
            j += 1




    return seq






