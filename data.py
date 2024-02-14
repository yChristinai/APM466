import pandas
import matplotlib.pyplot
import numpy as np
from scipy.optimize import newton

Jan8 = pandas.read_excel('APM466 Assignment1 bond data.xlsx', sheet_name='Jan8(Jan26)')
Jan9 = pandas.read_excel('APM466 Assignment1 bond data.xlsx', sheet_name='Jan.9')
Jan10 = pandas.read_excel('APM466 Assignment1 bond data.xlsx', sheet_name='Jan10')
Jan11 = pandas.read_excel('APM466 Assignment1 bond data.xlsx', sheet_name='Jan11')
Jan12 = pandas.read_excel('APM466 Assignment1 bond data.xlsx', sheet_name='Jan12(Jan29)')
Jan15 = pandas.read_excel('APM466 Assignment1 bond data.xlsx', sheet_name='Jan15')
Jan16 = pandas.read_excel('APM466 Assignment1 bond data.xlsx', sheet_name='Jan16')
Jan17 = pandas.read_excel('APM466 Assignment1 bond data.xlsx', sheet_name='Jan17')
Jan18 = pandas.read_excel('APM466 Assignment1 bond data.xlsx', sheet_name='Jan18')
Jan19 = pandas.read_excel('APM466 Assignment1 bond data.xlsx', sheet_name='Jan19')
lst_sheet = [Jan8, Jan9, Jan10, Jan11, Jan12, Jan15, Jan16, Jan17, Jan18, Jan19]
bond1, bond2, bond3 = [], [], []
bond4, bond5 = [], []
bond6, bond7, bond8 = [], [], []
bond9, bond10 = [],[]

for k in range(10):
    bond1 += [[[value] for value in lst_sheet[k].loc[6].values]] #10 bonds that I chose
    bond2 += [[[value] for value in lst_sheet[k].loc[10].values]]
    bond3 += [[[value] for value in lst_sheet[k].loc[16].values]]
    bond4 += [[[value] for value in lst_sheet[k].loc[20].values]]
    bond5 += [[[value] for value in lst_sheet[k].loc[23].values]]
    bond6 += [[[value] for value in lst_sheet[k].loc[24].values]]
    bond7 += [[[value] for value in lst_sheet[k].loc[28].values]]
    bond8 += [[[value] for value in lst_sheet[k].loc[29].values]]
    bond9 += [[[value] for value in lst_sheet[k].loc[31].values]]
    bond10 += [[[value] for value in lst_sheet[k].loc[32].values]]

face_v = 100
def pv_cal(bond, length):
    lst_pv = []
    lst_cp = []
    for i in range(length):
        coupon_rate = float(bond[i][2][0])/2
        coupon_payment = face_v * (coupon_rate/100)
        ai = coupon_rate * (i+1)
        pv = bond[i][6][0] + ai
        lst_cp.append(coupon_payment)
        lst_pv.append(pv)
    return lst_pv, lst_cp

bond1_pv, bond1_cp = pv_cal(bond1,1)[0], pv_cal(bond1,1)[1]
bond2_pv, bond2_cp = pv_cal(bond2,2)[0], pv_cal(bond2,2)[1]
bond3_pv, bond3_cp = pv_cal(bond3,3)[0], pv_cal(bond3,3)[1]
bond4_pv, bond4_cp = pv_cal(bond4,4)[0], pv_cal(bond4,4)[1]
bond5_pv, bond5_cp = pv_cal(bond5,5)[0], pv_cal(bond5,5)[1]
bond6_pv, bond6_cp = pv_cal(bond6,6)[0], pv_cal(bond6,6)[1]
bond7_pv, bond7_cp = pv_cal(bond7,7)[0], pv_cal(bond7,7)[1]
bond8_pv, bond8_cp = pv_cal(bond8,8)[0], pv_cal(bond8,8)[1]
bond9_pv, bond9_cp = pv_cal(bond9,9)[0], pv_cal(bond9,9)[1]
bond10_pv, bond10_cp = pv_cal(bond10,10)[0], pv_cal(bond9,10)[1]

def calculate_present_value(r, n, t, C, FV):
    return sum(C / (1 + r/n)**(i*n) for i in range(1, int(t*n)+1)) + FV / (1 + r/n)**(t*n)
def calculate_ytm(P, n, t, C, FV):
    ytm_function = lambda r: calculate_present_value(r, n, t, C, FV) - P
    ytm = newton(ytm_function, 0.05)
    return abs(ytm)
def calculate_ytm_iter(P, n, t, C, FV):
    lst_ytm = []
    for i in range(len(P)):
        lst_ytm.append(calculate_ytm(P[i], n, t, C[i], FV))
    return lst_ytm

ytm1 = calculate_ytm_iter(bond1_pv, 2, 0.5, bond1_cp, face_v)
ytm2 = calculate_ytm_iter(bond2_pv, 2, 1, bond2_cp, face_v)
ytm3 = calculate_ytm_iter(bond3_pv, 2, 1.5, bond3_cp, face_v)
ytm4 = calculate_ytm_iter(bond4_pv, 2, 2, bond4_cp, face_v)
ytm5 = calculate_ytm_iter(bond5_pv, 2, 2.5, bond5_cp, face_v)
ytm6 = calculate_ytm_iter(bond6_pv, 2, 3, bond6_cp, face_v)
ytm7 = calculate_ytm_iter(bond7_pv, 2, 3.5, bond7_cp, face_v)
ytm8 = calculate_ytm_iter(bond8_pv, 2, 4, bond8_cp, face_v)
ytm9 = calculate_ytm_iter(bond9_pv, 2, 4.5, bond9_cp, face_v)
ytm10 = calculate_ytm_iter(bond10_pv, 2, 5, bond10_cp, face_v)

def interpo(ytm1, ytm2): #interpolation of ytm
    if len(ytm1) < 10:
        empty = []
        for i in range(10-len(ytm1)):
            empty.append(ytm2[-(i+1)])
        for k in range(len(empty)):
            ytm1.append(empty[-(k+1)])
        return ytm1
    else:
        return ('invalid input, please try again')
ytm9_1 = interpo(ytm9, ytm10)
ytm8_1 = interpo(ytm8, ytm9_1)
ytm7_1 = interpo(ytm7, ytm8_1)
ytm6_1 = interpo(ytm6, ytm7_1)
ytm5_1 = interpo(ytm5, ytm6_1)
ytm4_1 = interpo(ytm4, ytm5_1)
ytm3_1 = interpo(ytm3, ytm4_1)
ytm2_1 = interpo(ytm2, ytm3_1)
ytm1_1 = interpo(ytm1, ytm2_1)

ts = ['2024-09-01', '2025-03-01', '2025-09-01','2026-03-01', '2026-09-01', '2027-03-01','2027-09-01', '2028-03-01', '2028-09-01','2029-03-01']

##matplotlib.pyplot.plot(ts, ytm1_1, marker='o', label = 'bond1')
##matplotlib.pyplot.plot(ts, ytm2_1, marker='o', label = 'bond2')
##matplotlib.pyplot.plot(ts, ytm3_1, marker='o', label = 'bond3')
##matplotlib.pyplot.plot(ts, ytm4_1, marker='o', label = 'bond4')
##matplotlib.pyplot.plot(ts, ytm5_1, marker='o', label = 'bond5')
##matplotlib.pyplot.plot(ts, ytm6_1, marker='o', label = 'bond6')
##matplotlib.pyplot.plot(ts, ytm7_1, marker='o', label = 'bond7')
##matplotlib.pyplot.plot(ts, ytm8_1, marker='o', label = 'bond8')
##matplotlib.pyplot.plot(ts, ytm9_1, marker='o', label = 'bond9')
##matplotlib.pyplot.plot(ts, ytm10, marker='o', label = 'bond10')
##matplotlib.pyplot.xlabel('Timestamp')
##matplotlib.pyplot.ylabel('Yield to Maturity (YTM)')
##matplotlib.pyplot.title('Yield curve for five years')
##matplotlib.pyplot.grid(True)
##matplotlib.pyplot.legend()
##matplotlib.pyplot.show()


#spot rate
def pv_spot(r, n, t, C, FV):
    return sum(C / (1 + r/n)**(i*n) for i in range(1, int(t*n))) + (C + FV) / (1 + r/n)**(t*n)

def calculate_sp(P, n, t, C, FV):
    spot_function = lambda r: pv_spot(r, n, t, C, FV) - P
    spot_rate = newton(spot_function, 0.05)
    return abs(spot_rate)
def calculate_spot_iter(P, n, t, C, FV):
    lst_spot = []
    for i in range(len(P)):
        lst_spot.append(calculate_ytm(P[i], n, t, C[i], FV))
    return lst_spot

sp1 = calculate_spot_iter(bond1_pv,2, 0.5, bond1_cp, face_v)
sp2 = calculate_spot_iter(bond2_pv, 2, 1, bond2_cp, face_v)
sp3 = calculate_spot_iter(bond3_pv, 2, 1.5, bond3_cp, face_v)
sp4 = calculate_spot_iter(bond4_pv, 2, 2, bond4_cp, face_v)
sp5 = calculate_spot_iter(bond5_pv, 2, 2.5, bond5_cp, face_v)
sp6 = calculate_spot_iter(bond6_pv, 2, 3, bond6_cp, face_v)
sp7 =calculate_spot_iter(bond7_pv, 2, 3.5, bond7_cp, face_v)
sp8 = calculate_spot_iter(bond8_pv, 2, 4, bond8_cp, face_v)
sp9 = calculate_spot_iter(bond9_pv, 2, 4.5, bond9_cp, face_v)
sp10 = calculate_spot_iter(bond10_pv, 2, 5, bond10_cp, face_v)

def interpo_sp(sp1, sp2): #interpolation of spot rate
    if len(sp1) < 10:
        empty = []
        for i in range(10-len(sp1)):
            empty.append(sp2[-(i+1)])
        for k in range(len(empty)):
            sp1.append(empty[-(k+1)])
        return sp1
    else:
        return ('invalid input, please try again')

sp9_1 = interpo_sp(sp9, sp10)
sp8_1 = interpo(sp8, sp9_1)
sp7_1 = interpo(sp7, sp8_1)
sp6_1 = interpo(sp6, sp7_1)
sp5_1 = interpo(sp5, sp6_1)
sp4_1 = interpo(sp4, sp5_1)
sp3_1 = interpo(sp3, sp4_1)
sp2_1 = interpo(sp2, sp3_1)
sp1_1 = interpo(sp1, sp2_1)

##matplotlib.pyplot.plot(ts, sp1_1, marker='o', label = 'bond1')
##matplotlib.pyplot.plot(ts, sp2_1, marker='o', label = 'bond2')
##matplotlib.pyplot.plot(ts, sp3_1, marker='o', label = 'bond3')
##matplotlib.pyplot.plot(ts, sp4_1, marker='o', label = 'bond4')
##matplotlib.pyplot.plot(ts, sp5_1, marker='o', label = 'bond5')
##matplotlib.pyplot.plot(ts, sp6_1, marker='o', label = 'bond6')
##matplotlib.pyplot.plot(ts, sp7_1, marker='o', label = 'bond7')
##matplotlib.pyplot.plot(ts, sp8_1, marker='o', label = 'bond8')
##matplotlib.pyplot.plot(ts, sp9_1, marker='o', label = 'bond9')
##matplotlib.pyplot.plot(ts, sp10, marker='o', label = 'bond10')
##matplotlib.pyplot.title('5-Year Spot Curve')
##matplotlib.pyplot.xlabel('Date')
##matplotlib.pyplot.ylabel('Spot Rate (%)')
##matplotlib.pyplot.grid(True)
##matplotlib.pyplot.legend()
##matplotlib.pyplot.show()

##Forward Rate
def cal_forward(spot_rate, start, end):
    lst_fr = []
    for i in range(end):
        if i != 0:
            forward = (spot_rate[2*(i+1)-1] * 2 * (i+1) - spot_rate[1])/2
            lst_fr.append(abs(forward))
    return lst_fr

fr_1 = cal_forward(sp1_1, 1, 5)
fr_2 = cal_forward(sp2_1, 1, 5)
fr_3 = cal_forward(sp3_1, 1, 5)
fr_4 = cal_forward(sp4_1, 1, 5)
fr_5 = cal_forward(sp5_1, 1, 5)
fr_6 = cal_forward(sp6_1, 1, 5)
fr_7 = cal_forward(sp7_1, 1, 5)
fr_8 = cal_forward(sp8_1, 1, 5)
fr_9 = cal_forward(sp9_1, 1, 5)
fr_10 = cal_forward(sp10, 1, 5)

time_slot = ['1-1 year', '1-2 year', '1-3 year', '1-4 year']

matplotlib.pyplot.plot(time_slot, fr_1, marker='o', label = 'bond1')
matplotlib.pyplot.plot(time_slot, fr_2, marker='o', label = 'bond2')
matplotlib.pyplot.plot(time_slot, fr_3, marker='o', label = 'bond3')
matplotlib.pyplot.plot(time_slot, fr_4, marker='o', label = 'bond4')
matplotlib.pyplot.plot(time_slot, fr_5, marker='o', label = 'bond5')
matplotlib.pyplot.plot(time_slot, fr_6, marker='o', label = 'bond6')
matplotlib.pyplot.plot(time_slot, fr_7, marker='o', label = 'bond7')
matplotlib.pyplot.plot(time_slot, fr_8, marker='o', label = 'bond8')
matplotlib.pyplot.plot(time_slot, fr_9, marker='o', label = 'bond9')
matplotlib.pyplot.plot(time_slot, fr_10, marker='o', label = 'bond10')
matplotlib.pyplot.title('Forward Rate Curve')
matplotlib.pyplot.xlabel('Time slot(year)')
matplotlib.pyplot.ylabel('Forward rate (%)')
matplotlib.pyplot.grid(True)
matplotlib.pyplot.legend()
##matplotlib.pyplot.show()

#Covariance
i = 5
j = 9
def log_equ(ytm):
    lst_log = []
    for i in range(10):
        if i != 0:
            temp = ytm[i]
            lst_log.append(abs(np.log(temp/ytm[i-1])))
    return lst_log
x1 = log_equ(ytm2_1)
x2 = log_equ(ytm4_1)
x3 = log_equ(ytm6_1)
x4 = log_equ(ytm8_1)
x5 = log_equ(ytm10)

def log_forward(fr):
    lst_fr = []
    for j in range(4):
        if j != 0:
            temp = fr[j]
            lst_fr.append(abs(np.log(temp/fr[j-1])))
    return lst_fr
fr1 = log_forward(fr_2)
fr2 = log_forward(fr_4)
fr3 = log_forward(fr_6)
fr4 = log_forward(fr_8)
fr5 = log_forward(fr_10)
print(fr5)



