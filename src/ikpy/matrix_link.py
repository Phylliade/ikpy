"""
Copyright (c) 2015, Xenomorphales
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of Aversive++ nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from . import link
import numpy as np
import sympy

class SubLink:
    def __init__(self, name):
        self.name = name

    def get_num_params(self):
        return 0

    def get_transformation_matrix(self):
        raise NotImplementedError

class ConstantMatrixSubLink(SubLink):
    def __init__(self, name, matrix):
        SubLink.__init__(self, name)
        if type(matrix) == np.matrix:
            self.matrix = matrix
        else:
            self.matrix = np.matrix(matrix)

    def get_transformation_matrix(self):
        return self.matrix

class VariableMatrixSubLink(SubLink):
    def __init__(self, name, matrix, symbols):
        SubLink.__init__(self, name)
        self.symbols = symbols
        self.matrix = matrix
        self.lambda_matrix = sympy.lambdify(self.symbols, sympy.Matrix(self.matrix), "numpy")
        self.sympy_lambda_matrix = sympy.lambdify(self.symbols, sympy.Matrix(self.matrix), "sympy")

    def get_num_params(self):
        return len(self.symbols)

    def get_transformation_matrix(self, *args):
        matrix = self.lambda_matrix
        if any(list(map(lambda e: type(e) == sympy.Symbol, args))):
            matrix = self.sympy_lambda_matrix
        if len(self.symbols) == len(args):
            return matrix(*args)
        else:
            raise IndexError("{} parameters for {} symbols".format(len(args), len(self.symbols)))

class MatrixLink(link.Link):
    def __init__(self, name, sublinks):
        link.Link.__init__(self, name, (None, None))
        self.sublinks = sublinks
        self._length = 1 #dirty
        sumparams = 0
        for l in sublinks:
            sumparams += l.get_num_params()
            if sumparams > 1:
                raise NotImplementedError("Current Link implementation don't support multiple params")

    def get_transformation_matrix(self, theta):
        ret = np.eye(4)
        for l in self.sublinks:
            if l.get_num_params() == 0:
                ret = ret * l.get_transformation_matrix()
            elif l.get_num_params() == 1:
                ret = ret * l.get_transformation_matrix(theta)
        if ret.shape == (4,1):
            tmp =  np.eye(4)
            tmp[0,3] = ret[0,0]
            tmp[1,3] = ret[1,0]
            tmp[2,3] = ret[2,0]
            tmp[3,3] = ret[3,0]
            return tmp
        return ret
