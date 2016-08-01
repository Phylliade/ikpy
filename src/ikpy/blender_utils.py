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

from ikpy import matrix_link
import sympy
import json
import subprocess
import os
import sys

def get_links_from_blender(blend_path, endpoint):
    python_script = sys.prefix + "/share/ikpy/blender_export.py"
    command = ["blender", blend_path, "--background", "--python", python_script]
    out = subprocess.Popen(command, stdout=subprocess.PIPE).communicate()[0]
    data = False
    out_data = []
    for l in out.split("\n"):
        if l == "end_data":
            data = False
        if data:
            out_data.append(l)
        if l == "begin_data":
            data = True
    json_links_list = json.loads("\n".join(out_data))
    json_links_dict = {}
    for l in json_links_list:
        json_links_dict[l["name"]] = l
    links_list = []
    while endpoint != None:
        json_link = json_links_dict[endpoint]
        if json_link["is_variable"]:
            link = matrix_link.VariableMatrixLink(json_link["name"], json_link["parent"], json_link["matrix"], [sympy.Symbol("x")])
            links_list = [link] + links_list
        else:
            link = matrix_link.ConstantMatrixLink(json_link["name"], json_link["parent"], json_link["matrix"])
            links_list = [link] + links_list
        endpoint = json_link["parent"]
    return links_list
