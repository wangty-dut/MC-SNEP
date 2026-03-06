import tensorflow.compat.v1 as tf
#define the third layer of LSTM
class Lstm31(object):
    def __init__(self, input_width, state_width, batch_size,time_weight,space_weight):
        self.input_width = input_width
        self.state_width = state_width
        self.batch_size = batch_size
        self.time_weight = time_weight
        self.space_weight = space_weight
        # initialize times
        self.times = 0
        # weight matrix of input gate
        self.Mikhi, self.Mikhk, self.Mikg = self.init_weight_mat()
        # weight matrix of output gate
        self.Mukhi, self.Mukhk, self.Mukg = self.init_weight_mat()
        # weight matrix of forget gate
        self.Mftkhi, self.Mftkhk, self.Mftkg = self.init_weight_mat()
        # weight matrix of forget gate
        self.Mfskhi, self.Mfskhk, self.Mfskg = self.init_weight_mat()

        self.Mijhi, self.Mijhj, self.Mijg = self.init_weight_mat()
        self.Mujhi, self.Mujhj, self.Mujg = self.init_weight_mat()
        self.Mftjhi, self.Mftjhj, self.Mftjg = self.init_weight_mat()
        self.Mfsjhi, self.Mfsjhj, self.Mfsjg = self.init_weight_mat()

        self.Milhi, self.Milhl, self.Milg = self.init_weight_mat()
        self.Mulhi, self.Mulhl, self.Mulg = self.init_weight_mat()
        self.Mftlhi, self.Mftlhl, self.Mftlg = self.init_weight_mat()
        self.Mfslhi, self.Mfslhl, self.Mfslg = self.init_weight_mat()

        self.Mimhi, self.Mimhm, self.Mimg = self.init_weight_mat()
        self.Mumhi, self.Mumhm, self.Mumg = self.init_weight_mat()
        self.Mftmhi, self.Mftmhm, self.Mftmg = self.init_weight_mat()
        self.Mfsmhi, self.Mfsmhm, self.Mfsmg = self.init_weight_mat()

        self.Minhi, self.Minhn, self.Ming = self.init_weight_mat()
        self.Munhi, self.Munhn, self.Mung = self.init_weight_mat()
        self.Mftnhi, self.Mftnhn, self.Mftng = self.init_weight_mat()
        self.Mfsnhi, self.Mfsnhn, self.Mfsng = self.init_weight_mat()

        self.Miohi, self.Mioho, self.Miog = self.init_weight_mat()
        self.Muohi, self.Muoho, self.Muog = self.init_weight_mat()
        self.Mftohi, self.Mftoho, self.Mftog = self.init_weight_mat()
        self.Mfsohi, self.Mfsoho, self.Mfsog = self.init_weight_mat()

        self.Mohi, self.Mohjk, self.Mog = self.init_weight_mat()

        self.itk_list = self.init_state_vec()
        self.utk_list = self.init_state_vec()
        self.ftk_list = self.init_state_vec()
        self.fsk_list = self.init_state_vec()
        self.ctk_list = self.init_state_vec()

        self.itj_list = self.init_state_vec()
        self.utj_list = self.init_state_vec()
        self.ftj_list = self.init_state_vec()
        self.fsj_list = self.init_state_vec()
        self.ctj_list = self.init_state_vec()

        self.itl_list = self.init_state_vec()
        self.utl_list = self.init_state_vec()
        self.ftl_list = self.init_state_vec()
        self.fsl_list = self.init_state_vec()
        self.ctl_list = self.init_state_vec()

        self.itm_list = self.init_state_vec()
        self.utm_list = self.init_state_vec()
        self.ftm_list = self.init_state_vec()
        self.fsm_list = self.init_state_vec()
        self.ctm_list = self.init_state_vec()

        self.itn_list = self.init_state_vec()
        self.utn_list = self.init_state_vec()
        self.ftn_list = self.init_state_vec()
        self.fsn_list = self.init_state_vec()
        self.ctn_list = self.init_state_vec()

        self.ito_list = self.init_state_vec()
        self.uto_list = self.init_state_vec()
        self.fto_list = self.init_state_vec()
        self.fso_list = self.init_state_vec()
        self.cto_list = self.init_state_vec()

        self.ct_list = self.init_state_vec()
        self.ot_list = self.init_state_vec()
        self.ht_list = self.init_state_vec()
        # control gate1 # 多门控
        self.onet=tf.zeros([self.batch_size, self.state_width])
        # control gate2
        self.twot=tf.zeros([self.batch_size, self.state_width])
        # control gate3
        self.threet = tf.zeros([self.batch_size, self.state_width])
        # control gate4
        self.fourt = tf.zeros([self.batch_size, self.state_width])
        # control gate5
        self.fivet = tf.zeros([self.batch_size, self.state_width])
        # control gate6
        self.sixt = tf.zeros([self.batch_size, self.state_width])

    def init_weight_mat(self):
        '''
        initialize weight matrix
        '''
        Mhi = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhk = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mg = tf.Variable(tf.random_normal([self.input_width, self.state_width], stddev=0.1))
        return Mhi,Mhk,Mg

    def init_weight_mat1(self):
        Mhi = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhk = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhj = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mg = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        return Mhi,Mhk,Mhj,Mg

    def init_state_vec(self):
        '''
        initialize hidden states
        '''
        state_vec_list = []
        state_vec_list.append(tf.zeros([self.batch_size, self.state_width]))
        return state_vec_list

    def calc_gate0(self,x,hi,hk,Mg,Mhi,Mhk):
        '''
        gate calculation
        '''
        net = tf.matmul(x, Mg) + tf.matmul(hi, Mhi) + tf.matmul(hk, Mhk)
        gate = tf.sigmoid(net)
        return gate
    def calc_gate00(self,x,hi,hk,Mg,Mhi,Mhk):
        net = tf.matmul(x, Mg) + tf.matmul(hi, Mhi) + tf.matmul(hk, Mhk)
        gate = tf.tanh(net)
        return gate


    def forward(self,hj_list,hk_list,hl_list, hm_list, hn_list, ho_list, x, control):
        '''
        forward calculation
        '''

        # proto_tensor = tf.make_tensor_proto(control)
        # data_numpy = tf.make_ndarray(proto_tensor)


        # data_numpy = tf.Session().run(control)
        # for i in range(self.batch_size):
        #     if data_numpy[i]==0:
        #         self.onet[i:i+1] = tf.ones([1, self.state_width])
        #     elif data_numpy[i] == 1:
        #         self.twot[i:i+1] = tf.ones([1, self.state_width])
        #     elif data_numpy[i] == 2:
        #         self.threet[i:i+1] = tf.ones([1, self.state_width])
        #     elif data_numpy[i] == 3:
        #         self.fourt[i:i+1] = tf.ones([1, self.state_width])
        #     elif data_numpy[i] == 4:
        #         self.fivet[i:i+1] = tf.ones([1, self.state_width])
        #     elif data_numpy[i] == 5:
        #         self.sixt[i:i+1] = tf.ones([1, self.state_width])
        # print(self.onet)

        self.onet = control[:, 0, :]
        self.twot = control[:, 1, :]
        self.threet = control[:, 2, :]
        self.fourt = control[:, 3, :]
        self.fivet = control[:, 4, :]
        self.sixt = control[:, 5, :]
        self.sevent = control[:, 6, :]
        self.eightt = control[:, 7, :]
        self.ninet = control[:, 8, :]

        # self.onet =tf.ones([1,self.state_width])
        # self.twot = tf.ones([1,self.state_width])
        # self.threet = tf.ones([1,self.state_width])
        # self.fourt = tf.ones([1,self.state_width])
        # self.fivet = tf.ones([1,self.state_width])
        # self.sixt = tf.ones([1,self.state_width])

        while self.times < 60:
            self.times += 1
            hi = self.ht_list[self.times - 1]
            hk = hk_list[self.times - 1]
            hj = hj_list[self.times - 1]
            hl = hl_list[self.times - 1]
            hm = hm_list[self.times - 1]
            hn = hn_list[self.times - 1]
            ho = ho_list[self.times - 1]
            # hp = hp_list[self.times - 1]
            # hq = hq_list[self.times - 1]
            # hr = hr_list[self.times - 1]

            # 时序1
            itk = self.calc_gate0(x[:, self.times-1, :],hi,hk,self.Mikg,self.Mikhi,self.Mikhk)
            self.itk_list.append(itk)

            utk = self.calc_gate00(x[:, self.times-1, :],hi,hk,self.Mukg,self.Mukhi,self.Mukhk)
            self.utk_list.append(utk)

            ftk = self.calc_gate0(x[:, self.times-1, :],hi,hk,self.Mftkg,self.Mftkhi,self.Mftkhk)
            self.ftk_list.append(ftk)

            fsk = self.calc_gate0(x[:, self.times-1, :],hi,hk,self.Mfskg,self.Mfskhi,self.Mfskhk)
            self.fsk_list.append(fsk)

            ctk = itk * utk + self.time_weight*ftk * self.ct_list[self.times - 1] + ftk * self.ctk_list[self.times - 1]
            self.ctk_list.append(ctk)

            # print(self.times)
            # 时序2
            itj = self.calc_gate0(x[:, self.times-1, :],hi,hj,self.Mijg,self.Mijhi,self.Mijhj)
            self.itj_list.append(itj)

            utj = self.calc_gate00(x[:, self.times-1, :],hi,hj,self.Mujg,self.Mujhi,self.Mujhj)
            self.utj_list.append(utk)

            ftj = self.calc_gate0(x[:, self.times-1, :],hi,hj,self.Mftjg,self.Mftjhi,self.Mftjhj)
            self.ftj_list.append(ftj)

            fsj = self.calc_gate0(x[:, self.times-1, :],hi,hj,self.Mfsjg,self.Mfsjhi,self.Mfsjhj)
            self.fsj_list.append(fsj)

            ctj = itj * utj + self.time_weight*ftj * self.ct_list[self.times - 1] + ftj * self.ctj_list[self.times - 1]
            self.ctj_list.append(ctj)

            # 时序3
            itl = self.calc_gate0(x[:, self.times-1, :], hi, hl, self.Milg, self.Milhi, self.Milhl)
            self.itl_list.append(itl)

            utl = self.calc_gate00(x[:, self.times-1, :], hi, hl, self.Mulg, self.Mulhi, self.Mulhl)
            self.utl_list.append(utk)

            ftl = self.calc_gate0(x[:, self.times-1, :], hi, hl, self.Mftlg, self.Mftlhi, self.Mftlhl)
            self.ftl_list.append(ftl)

            fsl = self.calc_gate0(x[:, self.times-1, :], hi, hl, self.Mfslg, self.Mfslhi, self.Mfslhl)
            self.fsl_list.append(fsl)

            ctl = itl * utl + self.time_weight * ftl * self.ct_list[self.times - 1] + ftl * self.ctl_list[self.times - 1]
            self.ctl_list.append(ctl)

            # 时序4
            itm = self.calc_gate0(x[:, self.times-1, :], hi, hm, self.Mimg, self.Mimhi, self.Mimhm)
            self.itm_list.append(itm)

            utm = self.calc_gate00(x[:, self.times-1, :], hi, hm, self.Mumg, self.Mumhi, self.Mumhm)
            self.utm_list.append(utk)

            ftm = self.calc_gate0(x[:, self.times-1, :], hi, hm, self.Mftmg, self.Mftmhi, self.Mftmhm)
            self.ftm_list.append(ftm)

            fsm = self.calc_gate0(x[:, self.times-1, :], hi, hm, self.Mfsmg, self.Mfsmhi, self.Mfsmhm)
            self.fsm_list.append(fsm)

            ctm = itm * utm + self.time_weight * ftm * self.ct_list[self.times - 1] + ftm * self.ctm_list[self.times - 1]
            self.ctm_list.append(ctm)

            # 时序5
            itn = self.calc_gate0(x[:, self.times-1, :], hi, hn, self.Ming, self.Minhi, self.Minhn)
            self.itn_list.append(itn)

            utn = self.calc_gate00(x[:, self.times-1, :], hi, hn, self.Mung, self.Munhi, self.Munhn)
            self.utn_list.append(utk)

            ftn = self.calc_gate0(x[:, self.times-1, :], hi, hn, self.Mftng, self.Mftnhi, self.Mftnhn)
            self.ftn_list.append(ftn)

            fsn = self.calc_gate0(x[:, self.times-1, :], hi, hn, self.Mfsng, self.Mfsnhi, self.Mfsnhn)
            self.fsn_list.append(fsn)

            ctn = itn * utn + self.time_weight * ftn * self.ct_list[self.times - 1] + ftn * self.ctn_list[self.times - 1]
            self.ctn_list.append(ctn)

            # 时序6
            ito = self.calc_gate0(x[:, self.times-1, :], hi, ho, self.Miog, self.Miohi, self.Mioho)
            self.ito_list.append(ito)

            uto = self.calc_gate00(x[:, self.times-1, :], hi, ho, self.Muog, self.Muohi, self.Muoho)
            self.uto_list.append(utk)

            fto = self.calc_gate0(x[:, self.times-1, :], hi, ho, self.Mftog, self.Mftohi, self.Mftoho)
            self.fto_list.append(fto)

            fso = self.calc_gate0(x[:, self.times-1, :], hi, ho, self.Mfsog, self.Mfsohi, self.Mfsoho)
            self.fso_list.append(fso)

            cto = ito * uto + self.time_weight * fto * self.ct_list[self.times - 1] + fto * self.cto_list[self.times - 1]
            self.cto_list.append(cto)

            # # 时序6
            # ito = self.calc_gate0(x[:, self.times - 1, :], hi, ho, self.Miog, self.Miohi, self.Mioho)
            # self.ito_list.append(ito)
            # 
            # uto = self.calc_gate00(x[:, self.times - 1, :], hi, ho, self.Muog, self.Muohi, self.Muoho)
            # self.uto_list.append(utk)
            # 
            # fto = self.calc_gate0(x[:, self.times - 1, :], hi, ho, self.Mftog, self.Mftohi, self.Mftoho)
            # self.fto_list.append(fto)
            # 
            # fso = self.calc_gate0(x[:, self.times - 1, :], hi, ho, self.Mfsog, self.Mfsohi, self.Mfsoho)
            # self.fso_list.append(fso)
            # 
            # cto = ito * uto + self.time_weight * fto * self.ct_list[self.times - 1] + fto * self.cto_list[
            #     self.times - 1]
            # self.cto_list.append(cto)

            htij=self.space_weight*(self.onet*hj + self.twot*hk + self.threet*hl + self.fourt*hm + self.fivet*hn + self.sixt*ho)

            ct =self.space_weight*(self.onet*ctj + self.twot*ctk + self.threet*ctl+self.fourt*ctm + self.fivet*ctn + self.sixt*cto)
            self.ct_list.append(ct)

            ot =tf.sigmoid(tf.matmul(x[:, self.times-1, :], self.Mog)+tf.matmul(hi, self.Mohi)+tf.matmul(htij, self.Mohjk))

            ht = ot * tf.tanh(ct)
            self.ht_list.append(ht)
        return self.ht_list[-1], self.threet

class Lstm32(object):
    def __init__(self, input_width, state_width, batch_size, time_weight, space_weight):
        self.input_width = input_width
        self.state_width = state_width
        self.batch_size = batch_size
        self.time_weight = time_weight
        self.space_weight = space_weight
        # initialize times
        self.times = 0
        # weight matrix of input gate
        self.Mikhi, self.Mikhk, self.Mikg = self.init_weight_mat()
        # weight matrix of output gate
        self.Mukhi, self.Mukhk, self.Mukg = self.init_weight_mat()
        # weight matrix of forget gate
        self.Mftkhi, self.Mftkhk, self.Mftkg = self.init_weight_mat()
        # weight matrix of forget gate
        self.Mfskhi, self.Mfskhk, self.Mfskg = self.init_weight_mat()

        self.Mijhi, self.Mijhj, self.Mijg = self.init_weight_mat()
        self.Mujhi, self.Mujhj, self.Mujg = self.init_weight_mat()
        self.Mftjhi, self.Mftjhj, self.Mftjg = self.init_weight_mat()
        self.Mfsjhi, self.Mfsjhj, self.Mfsjg = self.init_weight_mat()

        self.Milhi, self.Milhl, self.Milg = self.init_weight_mat()
        self.Mulhi, self.Mulhl, self.Mulg = self.init_weight_mat()
        self.Mftlhi, self.Mftlhl, self.Mftlg = self.init_weight_mat()
        self.Mfslhi, self.Mfslhl, self.Mfslg = self.init_weight_mat()

        self.Mimhi, self.Mimhm, self.Mimg = self.init_weight_mat()
        self.Mumhi, self.Mumhm, self.Mumg = self.init_weight_mat()
        self.Mftmhi, self.Mftmhm, self.Mftmg = self.init_weight_mat()
        self.Mfsmhi, self.Mfsmhm, self.Mfsmg = self.init_weight_mat()

        self.Minhi, self.Minhn, self.Ming = self.init_weight_mat()
        self.Munhi, self.Munhn, self.Mung = self.init_weight_mat()
        self.Mftnhi, self.Mftnhn, self.Mftng = self.init_weight_mat()
        self.Mfsnhi, self.Mfsnhn, self.Mfsng = self.init_weight_mat()

        self.Miohi, self.Mioho, self.Miog = self.init_weight_mat()
        self.Muohi, self.Muoho, self.Muog = self.init_weight_mat()
        self.Mftohi, self.Mftoho, self.Mftog = self.init_weight_mat()
        self.Mfsohi, self.Mfsoho, self.Mfsog = self.init_weight_mat()

        self.Miphi, self.Miphp, self.Mipg = self.init_weight_mat()
        self.Muphi, self.Muphp, self.Mupg = self.init_weight_mat()
        self.Mftphi, self.Mftphp, self.Mftpg = self.init_weight_mat()
        self.Mfsphi, self.Mfsphp, self.Mfspg = self.init_weight_mat()

        self.Miqhi, self.Miqhq, self.Miqg = self.init_weight_mat()
        self.Muqhi, self.Muqhq, self.Muqg = self.init_weight_mat()
        self.Mftqhi, self.Mftqhq, self.Mftqg = self.init_weight_mat()
        self.Mfsqhi, self.Mfsqhq, self.Mfsqg = self.init_weight_mat()

        self.Mirhi, self.Mirhr, self.Mirg = self.init_weight_mat()
        self.Murhi, self.Murhr, self.Murg = self.init_weight_mat()
        self.Mftrhi, self.Mftrhr, self.Mftrg = self.init_weight_mat()
        self.Mfsrhi, self.Mfsrhr, self.Mfsrg = self.init_weight_mat()

        self.Mohi, self.Mohjk, self.Mog = self.init_weight_mat()

        self.itk_list = self.init_state_vec()
        self.utk_list = self.init_state_vec()
        self.ftk_list = self.init_state_vec()
        self.fsk_list = self.init_state_vec()
        self.ctk_list = self.init_state_vec()

        self.itj_list = self.init_state_vec()
        self.utj_list = self.init_state_vec()
        self.ftj_list = self.init_state_vec()
        self.fsj_list = self.init_state_vec()
        self.ctj_list = self.init_state_vec()

        self.itl_list = self.init_state_vec()
        self.utl_list = self.init_state_vec()
        self.ftl_list = self.init_state_vec()
        self.fsl_list = self.init_state_vec()
        self.ctl_list = self.init_state_vec()

        self.itm_list = self.init_state_vec()
        self.utm_list = self.init_state_vec()
        self.ftm_list = self.init_state_vec()
        self.fsm_list = self.init_state_vec()
        self.ctm_list = self.init_state_vec()

        self.itn_list = self.init_state_vec()
        self.utn_list = self.init_state_vec()
        self.ftn_list = self.init_state_vec()
        self.fsn_list = self.init_state_vec()
        self.ctn_list = self.init_state_vec()

        self.ito_list = self.init_state_vec()
        self.uto_list = self.init_state_vec()
        self.fto_list = self.init_state_vec()
        self.fso_list = self.init_state_vec()
        self.cto_list = self.init_state_vec()

        self.itp_list = self.init_state_vec()
        self.utp_list = self.init_state_vec()
        self.ftp_list = self.init_state_vec()
        self.fsp_list = self.init_state_vec()
        self.ctp_list = self.init_state_vec()

        self.itq_list = self.init_state_vec()
        self.utq_list = self.init_state_vec()
        self.ftq_list = self.init_state_vec()
        self.fsq_list = self.init_state_vec()
        self.ctq_list = self.init_state_vec()

        self.itr_list = self.init_state_vec()
        self.utr_list = self.init_state_vec()
        self.ftr_list = self.init_state_vec()
        self.fsr_list = self.init_state_vec()
        self.ctr_list = self.init_state_vec()

        self.ct_list = self.init_state_vec()
        self.ot_list = self.init_state_vec()
        self.ht_list = self.init_state_vec()
        # control gate1
        self.onet = tf.zeros([self.batch_size, self.state_width])
        # control gate2
        self.twot = tf.zeros([self.batch_size, self.state_width])
        # control gate3
        self.threet = tf.zeros([self.batch_size, self.state_width])
        # control gate4
        self.fourt = tf.zeros([self.batch_size, self.state_width])
        # control gate5
        self.fivet = tf.zeros([self.batch_size, self.state_width])
        # control gate6
        self.sixt = tf.zeros([self.batch_size, self.state_width])

    def init_weight_mat(self):
        '''
        initialize weight matrix
        '''
        Mhi = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhk = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mg = tf.Variable(tf.random_normal([self.input_width, self.state_width], stddev=0.1))
        return Mhi, Mhk, Mg

    def init_weight_mat1(self):
        Mhi = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhk = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhj = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mg = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        return Mhi, Mhk, Mhj, Mg

    def init_state_vec(self):
        '''
        initialize hidden states
        '''
        state_vec_list = []
        state_vec_list.append(tf.zeros([self.batch_size, self.state_width]))
        return state_vec_list

    def calc_gate0(self, x, hi, hk, Mg, Mhi, Mhk):
        '''
        gate calculation
        '''
        net = tf.matmul(x, Mg) + tf.matmul(hi, Mhi) + tf.matmul(hk, Mhk)
        gate = tf.sigmoid(net)
        return gate

    def calc_gate00(self, x, hi, hk, Mg, Mhi, Mhk):
        net = tf.matmul(x, Mg) + tf.matmul(hi, Mhi) + tf.matmul(hk, Mhk)
        gate = tf.tanh(net)
        return gate

    def forward(self, hj_list, hk_list, hl_list, hm_list, hn_list, ho_list, hp_list, hq_list, hr_list, x, control):
        '''
        forward calculation
        '''

        # proto_tensor = tf.make_tensor_proto(control)
        # data_numpy = tf.make_ndarray(proto_tensor)

        # data_numpy = tf.Session().run(control)
        # for i in range(self.batch_size):
        #     if data_numpy[i]==0:
        #         self.onet[i:i+1] = tf.ones([1, self.state_width])
        #     elif data_numpy[i] == 1:
        #         self.twot[i:i+1] = tf.ones([1, self.state_width])
        #     elif data_numpy[i] == 2:
        #         self.threet[i:i+1] = tf.ones([1, self.state_width])
        #     elif data_numpy[i] == 3:
        #         self.fourt[i:i+1] = tf.ones([1, self.state_width])
        #     elif data_numpy[i] == 4:
        #         self.fivet[i:i+1] = tf.ones([1, self.state_width])
        #     elif data_numpy[i] == 5:
        #         self.sixt[i:i+1] = tf.ones([1, self.state_width])
        # print(self.onet)

        self.onet = control[:, 0, :]
        self.twot = control[:, 1, :]
        self.threet = control[:, 2, :]
        self.fourt = control[:, 3, :]
        self.fivet = control[:, 4, :]
        self.sixt = control[:, 5, :]
        self.sevent = control[:, 6, :]
        self.eightt = control[:, 7, :]
        self.ninet = control[:, 8, :]

        # self.onet =tf.ones([1,self.state_width])
        # self.twot = tf.ones([1,self.state_width])
        # self.threet = tf.ones([1,self.state_width])
        # self.fourt = tf.ones([1,self.state_width])
        # self.fivet = tf.ones([1,self.state_width])
        # self.sixt = tf.ones([1,self.state_width])

        while self.times < 60:
            self.times += 1
            hi = self.ht_list[self.times - 1]
            hk = hk_list[self.times - 1]
            hj = hj_list[self.times - 1]
            hl = hl_list[self.times - 1]
            hm = hm_list[self.times - 1]
            hn = hn_list[self.times - 1]
            ho = ho_list[self.times - 1]
            hp = hp_list[self.times - 1]
            hq = hq_list[self.times - 1]
            hr = hr_list[self.times - 1]

            # 时序1
            itk = self.calc_gate0(x[:, self.times - 1, :], hi, hk, self.Mikg, self.Mikhi, self.Mikhk)
            self.itk_list.append(itk)

            utk = self.calc_gate00(x[:, self.times - 1, :], hi, hk, self.Mukg, self.Mukhi, self.Mukhk)
            self.utk_list.append(utk)

            ftk = self.calc_gate0(x[:, self.times - 1, :], hi, hk, self.Mftkg, self.Mftkhi, self.Mftkhk)
            self.ftk_list.append(ftk)

            fsk = self.calc_gate0(x[:, self.times - 1, :], hi, hk, self.Mfskg, self.Mfskhi, self.Mfskhk)
            self.fsk_list.append(fsk)

            ctk = itk * utk + self.time_weight * ftk * self.ct_list[self.times - 1] + ftk * self.ctk_list[
                self.times - 1]
            self.ctk_list.append(ctk)

            # print(self.times)
            # 时序2
            itj = self.calc_gate0(x[:, self.times - 1, :], hi, hj, self.Mijg, self.Mijhi, self.Mijhj)
            self.itj_list.append(itj)

            utj = self.calc_gate00(x[:, self.times - 1, :], hi, hj, self.Mujg, self.Mujhi, self.Mujhj)
            self.utj_list.append(utk)

            ftj = self.calc_gate0(x[:, self.times - 1, :], hi, hj, self.Mftjg, self.Mftjhi, self.Mftjhj)
            self.ftj_list.append(ftj)

            fsj = self.calc_gate0(x[:, self.times - 1, :], hi, hj, self.Mfsjg, self.Mfsjhi, self.Mfsjhj)
            self.fsj_list.append(fsj)

            ctj = itj * utj + self.time_weight * ftj * self.ct_list[self.times - 1] + ftj * self.ctj_list[
                self.times - 1]
            self.ctj_list.append(ctj)

            # 时序3
            itl = self.calc_gate0(x[:, self.times - 1, :], hi, hl, self.Milg, self.Milhi, self.Milhl)
            self.itl_list.append(itl)

            utl = self.calc_gate00(x[:, self.times - 1, :], hi, hl, self.Mulg, self.Mulhi, self.Mulhl)
            self.utl_list.append(utk)

            ftl = self.calc_gate0(x[:, self.times - 1, :], hi, hl, self.Mftlg, self.Mftlhi, self.Mftlhl)
            self.ftl_list.append(ftl)

            fsl = self.calc_gate0(x[:, self.times - 1, :], hi, hl, self.Mfslg, self.Mfslhi, self.Mfslhl)
            self.fsl_list.append(fsl)

            ctl = itl * utl + self.time_weight * ftl * self.ct_list[self.times - 1] + ftl * self.ctl_list[
                self.times - 1]
            self.ctl_list.append(ctl)

            # 时序4
            itm = self.calc_gate0(x[:, self.times - 1, :], hi, hm, self.Mimg, self.Mimhi, self.Mimhm)
            self.itm_list.append(itm)

            utm = self.calc_gate00(x[:, self.times - 1, :], hi, hm, self.Mumg, self.Mumhi, self.Mumhm)
            self.utm_list.append(utk)

            ftm = self.calc_gate0(x[:, self.times - 1, :], hi, hm, self.Mftmg, self.Mftmhi, self.Mftmhm)
            self.ftm_list.append(ftm)

            fsm = self.calc_gate0(x[:, self.times - 1, :], hi, hm, self.Mfsmg, self.Mfsmhi, self.Mfsmhm)
            self.fsm_list.append(fsm)

            ctm = itm * utm + self.time_weight * ftm * self.ct_list[self.times - 1] + ftm * self.ctm_list[
                self.times - 1]
            self.ctm_list.append(ctm)

            # 时序5
            itn = self.calc_gate0(x[:, self.times - 1, :], hi, hn, self.Ming, self.Minhi, self.Minhn)
            self.itn_list.append(itn)

            utn = self.calc_gate00(x[:, self.times - 1, :], hi, hn, self.Mung, self.Munhi, self.Munhn)
            self.utn_list.append(utk)

            ftn = self.calc_gate0(x[:, self.times - 1, :], hi, hn, self.Mftng, self.Mftnhi, self.Mftnhn)
            self.ftn_list.append(ftn)

            fsn = self.calc_gate0(x[:, self.times - 1, :], hi, hn, self.Mfsng, self.Mfsnhi, self.Mfsnhn)
            self.fsn_list.append(fsn)

            ctn = itn * utn + self.time_weight * ftn * self.ct_list[self.times - 1] + ftn * self.ctn_list[
                self.times - 1]
            self.ctn_list.append(ctn)

            # 时序6
            ito = self.calc_gate0(x[:, self.times - 1, :], hi, ho, self.Miog, self.Miohi, self.Mioho)
            self.ito_list.append(ito)

            uto = self.calc_gate00(x[:, self.times - 1, :], hi, ho, self.Muog, self.Muohi, self.Muoho)
            self.uto_list.append(utk)

            fto = self.calc_gate0(x[:, self.times - 1, :], hi, ho, self.Mftog, self.Mftohi, self.Mftoho)
            self.fto_list.append(fto)

            fso = self.calc_gate0(x[:, self.times - 1, :], hi, ho, self.Mfsog, self.Mfsohi, self.Mfsoho)
            self.fso_list.append(fso)

            cto = ito * uto + self.time_weight * fto * self.ct_list[self.times - 1] + fto * self.cto_list[
                self.times - 1]
            self.cto_list.append(cto)

            # 时序7
            itp = self.calc_gate0(x[:, self.times - 1, :], hi, hp, self.Mipg, self.Miphi, self.Miphp)
            self.itp_list.append(itp)

            utp = self.calc_gate00(x[:, self.times - 1, :], hi, hp, self.Mupg, self.Muphi, self.Muphp)
            self.utp_list.append(utk)

            ftp = self.calc_gate0(x[:, self.times - 1, :], hi, hp, self.Mftpg, self.Mftphi, self.Mftphp)
            self.ftp_list.append(ftp)

            fsp = self.calc_gate0(x[:, self.times - 1, :], hi, hp, self.Mfspg, self.Mfsphi, self.Mfsphp)
            self.fsp_list.append(fsp)

            ctp = itp * utp + self.time_weight * ftp * self.ct_list[self.times - 1] + ftp * self.ctp_list[
                self.times - 1]
            self.ctp_list.append(ctp)
            
            # 时序8
            itq = self.calc_gate0(x[:, self.times - 1, :], hi, hq, self.Miqg, self.Miqhi, self.Miqhq)
            self.itq_list.append(itq)

            utq = self.calc_gate00(x[:, self.times - 1, :], hi, hq, self.Muqg, self.Muqhi, self.Muqhq)
            self.utq_list.append(utk)

            ftq = self.calc_gate0(x[:, self.times - 1, :], hi, hq, self.Mftqg, self.Mftqhi, self.Mftqhq)
            self.ftq_list.append(ftq)

            fsq = self.calc_gate0(x[:, self.times - 1, :], hi, hq, self.Mfsqg, self.Mfsqhi, self.Mfsqhq)
            self.fsq_list.append(fsq)

            ctq = itq * utq + self.time_weight * ftq * self.ct_list[self.times - 1] + ftq * self.ctq_list[
                self.times - 1]
            self.ctq_list.append(ctq)
            
            # 时序9
            itr = self.calc_gate0(x[:, self.times - 1, :], hi, hr, self.Mirg, self.Mirhi, self.Mirhr)
            self.itr_list.append(itr)

            utr = self.calc_gate00(x[:, self.times - 1, :], hi, hr, self.Murg, self.Murhi, self.Murhr)
            self.utr_list.append(utk)

            ftr = self.calc_gate0(x[:, self.times - 1, :], hi, hr, self.Mftrg, self.Mftrhi, self.Mftrhr)
            self.ftr_list.append(ftr)

            fsr = self.calc_gate0(x[:, self.times - 1, :], hi, hr, self.Mfsrg, self.Mfsrhi, self.Mfsrhr)
            self.fsr_list.append(fsr)

            ctr = itr * utr + self.time_weight * ftr * self.ct_list[self.times - 1] + ftr * self.ctr_list[
                self.times - 1]
            self.ctr_list.append(ctr)

            htij = self.space_weight * (
                        self.onet * hj + self.twot * hk + self.threet * hl + self.fourt * hm + self.fivet * hn + self.sixt * ho + self.sevent * hp + self.eightt * hq + self.ninet * hr)

            ct = self.space_weight * (
                        self.onet * ctj + self.twot * ctk + self.threet * ctl + self.fourt * ctm + self.fivet * ctn + self.sixt * cto + self.sevent * ctp + self.eightt * ctq + self.ninet * ctr)
            self.ct_list.append(ct)

            ot = tf.sigmoid(
                tf.matmul(x[:, self.times - 1, :], self.Mog) + tf.matmul(hi, self.Mohi) + tf.matmul(htij, self.Mohjk))

            ht = ot * tf.tanh(ct)
            self.ht_list.append(ht)
        return self.ht_list[-1], self.threet
    

class Lstm33(object):
    def __init__(self, input_width, state_width, batch_size,time_weight,space_weight):
        self.input_width = input_width
        self.state_width = state_width
        self.batch_size = batch_size
        self.time_weight = time_weight
        self.space_weight = space_weight
        self.times = 0

        self.Mikhi, self.Mikhk, self.Mikg = self.init_weight_mat()
        self.Mukhi, self.Mukhk, self.Mukg = self.init_weight_mat()
        self.Mftkhi, self.Mftkhk, self.Mftkg = self.init_weight_mat()
        self.Mfskhi, self.Mfskhk, self.Mfskg = self.init_weight_mat()

        self.Mijhi, self.Mijhj, self.Mijg = self.init_weight_mat()
        self.Mujhi, self.Mujhj, self.Mujg = self.init_weight_mat()
        self.Mftjhi, self.Mftjhj, self.Mftjg = self.init_weight_mat()
        self.Mfsjhi, self.Mfsjhj, self.Mfsjg = self.init_weight_mat()

        self.Mohi, self.Mohjk, self.Mog = self.init_weight_mat()

        self.itk_list = self.init_state_vec()
        self.utk_list = self.init_state_vec()
        self.ftk_list = self.init_state_vec()
        self.fsk_list = self.init_state_vec()
        self.ctk_list = self.init_state_vec()

        self.itj_list = self.init_state_vec()
        self.utj_list = self.init_state_vec()
        self.ftj_list = self.init_state_vec()
        self.fsj_list = self.init_state_vec()
        self.ctj_list = self.init_state_vec()

        self.ct_list = self.init_state_vec()
        self.ot_list = self.init_state_vec()
        self.ht_list = self.init_state_vec()
        self.tt=tf.ones([self.state_width, self.batch_size])
        self.ttt=tf.zeros([self.state_width, self.batch_size])

    def init_weight_mat(self):
        Mhi = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhk = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mg = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        return Mhi,Mhk,Mg

    def init_weight_mat1(self):
        Mhi = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhk = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mhj = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        Mg = tf.Variable(tf.random_normal([self.state_width, self.state_width], stddev=0.1))
        return Mhi,Mhk,Mhj,Mg

    def init_state_vec(self):
        state_vec_list = []
        state_vec_list.append(tf.zeros([self.state_width, self.batch_size]))
        return state_vec_list

    def calc_gate0(self,x,hi,hk,Mg,Mhi,Mhk):
        net = tf.matmul(Mg,x) + tf.matmul(Mhi,hi) + tf.matmul(Mhk,hk)
        gate = tf.sigmoid(net)
        return gate
    def calc_gate00(self,x,hi,hk,Mg,Mhi,Mhk):
        net = tf.matmul(Mg,x) + tf.matmul(Mhi,hi) + tf.matmul(Mhk,hk)
        gate = tf.tanh(net)
        return gate


    def forward(self,hk,hj,x):
        self.times += 1
        hi = self.ht_list[self.times - 1]
        itk = self.calc_gate0(x,hi,hk,self.Mikg,self.Mikhi,self.Mikhk)
        self.itk_list.append(itk)
        utk = self.calc_gate00(x,hi,hk,self.Mukg,self.Mukhi,self.Mukhk)
        self.utk_list.append(utk)
        ftk = self.calc_gate0(x,hi,hk,self.Mftkg,self.Mftkhi,self.Mftkhk)
        self.ftk_list.append(ftk)
        fsk = self.calc_gate0(x,hi,hk,self.Mfskg,self.Mfskhi,self.Mfskhk)
        self.fsk_list.append(fsk)
        ctk = itk * utk + self.time_weight*ftk * self.ct_list[self.times - 1] + ftk * self.ctk_list[self.times - 1]
        self.ctk_list.append(ctk)

        itj = self.calc_gate0(x,hi,hj,self.Mijg,self.Mijhi,self.Mijhj)
        self.itj_list.append(itj)
        utj = self.calc_gate00(x,hi,hj,self.Mujg,self.Mujhi,self.Mujhj)
        self.utj_list.append(utk)
        ftj = self.calc_gate0(x,hi,hj,self.Mftjg,self.Mftjhi,self.Mftjhj)
        self.ftj_list.append(ftj)
        fsj = self.calc_gate0(x,hi,hj,self.Mfsjg,self.Mfsjhi,self.Mfsjhj)
        self.fsj_list.append(fsj)
        ctj = itj * utj + self.time_weight*ftj * self.ct_list[self.times - 1] + ftj * self.ctj_list[self.times - 1]

        htij=self.space_weight*(self.tt*hj + self.ttt*hk)
        ct =self.space_weight*(self.tt*ctj + self.ttt*ctk)
        self.ct_list.append(ct)
        ot =tf.sigmoid(tf.matmul(self.Mog, x)+tf.matmul(self.Mohi,hi)+tf.matmul(self.Mohjk, htij))
        ht = ot * tf.tanh(ct)
        self.ht_list.append(ht)
        return ht