import numpy as np
import sympy as sp
from functools import reduce

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library.standard_gates import XGate
from qiskit.extensions import UnitaryGate


def Unitary_decompose_2lvl(U,U_s=None,iter_n=0):
    e = sp.eye(iter_n)
    if U_s is None:
        U_s = []
        
    for i in range(1,U.shape[0]):
        U_i = sp.eye(U.shape[0],dtype = 'complex_')
        d = sp.sqrt(sp.Abs(U[0,0])**2+sp.Abs(U[i,0])**2)
        U_i[0,0],U_i[0,i],U_i[i,0],U_i[i,i] = sp.conjugate(U[0,0])/d, sp.conjugate(U[i,0])/d, U[i,0]/d, -1*U[0,0]/d
        U_i = sp.simplify(U_i)
        U = U_i*U
        U_i = sp.diag(e,U_i)
        U_s.append(U_i)
        
    if np.all(np.allclose(np.array(U).astype(np.clongdouble),np.eye(U.shape[0],dtype = 'complex_'))):
        return U_s
    else:
        if U.shape[0]==3:
            U = sp.simplify(U)
            U_i = sp.conjugate(U.T)
            U_i = sp.diag(e,U_i)
            U_s.append(U_i)
            return U_s
        else:
            iter_n+=1
            U = U[1:,1:]
            Unitary_decompose_2lvl(U,U_s,iter_n)
            return U_s

def Gray_code(start,end):
    g = []
    g.append(start)
    for i, symb in enumerate(zip(start,end)):
        if symb[0] != symb[1]:
            g.append(g[-1][0:i]+symb[1]+g[-1][i+1:])
        else:
            continue
        
    return g



def circuit_2lvl(U):
    U_np = np.array(U) # dtype=np.clongdouble
    num_qubits = int(np.log2(U_np.shape[0]))
    unique, counts = np.unique(np.nonzero(U_np)[0], return_counts=True)
    acting_vecs = unique[counts == 2]
    U_reduced = np.array([ [U_np[acting_vecs[0]][acting_vecs[0]],U_np[acting_vecs[0]][acting_vecs[1]]], 
                           [U_np[acting_vecs[1]][acting_vecs[0]],U_np[acting_vecs[1]][acting_vecs[1]]]])
    acting_vecs_bin=[]
    
    for v in acting_vecs:
        x = bin(v).split('b')[1]
        if x != num_qubits:
            acting_vecs_bin.append((num_qubits-len(x))*'0'+x)
    Gray_seq = Gray_code(acting_vecs_bin[0],acting_vecs_bin[1])
    qr = QuantumRegister(num_qubits,'q')
    qc_control = QuantumCircuit(qr)
    qc_reduced_unitary = QuantumCircuit(qr)
    qc = QuantumCircuit(qr)
    for i, seq in enumerate(Gray_seq[:-2]):
        target = [i for i,(c1,c2) in enumerate(zip(seq,Gray_seq[i+1])) if c1!=c2]
        controled = [i for i in range(num_qubits) if i!=target[0]]
        control_state  = Gray_seq[i+1][:target[0]]+ Gray_seq[i+1][target[0]+1:]
        cx_gate = XGate().control(num_qubits-1,ctrl_state=control_state)
        qc_control.append(cx_gate, controled+target)
    U_reduced_operator = UnitaryGate(U_reduced,label='U').control(num_qubits-1)
    qc_reduced_unitary.append(U_reduced_operator,range(3))
    
    qc.append(qc_control, range(3))
    qc.append(qc_reduced_unitary, range(3))
    qc.append(qc_control.inverse(), range(3))
    
    
    display(qc.decompose().draw('mpl'))
    return qc.decompose()