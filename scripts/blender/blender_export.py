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

import bpy
import json

def bone_matrix(bone):
    if bone.parent == None:
        return bone.matrix_local
    else:
        return bone.parent.matrix_local.inverted() * bone.matrix_local
    
def find(bone, func):
    func(bone)
    for b in bone.children:
        find(b, func)
      
def real2cas(value):
    ret  = "CAS::General<double, double>::"
    ret += "Const::"
    p = int(value*1024)
    q = int(1024)
    if(p%q == 0):
        ret += "Integer<"+str(int(p/q))+">"
    else:
        ret += "Rational<"+str(p)+","+str(q)+">"
    return ret

def identity2cas():
    return "CAS::General<double, double>::Identity"

def matrix2cas(mat):
    ret = []
    for row in mat:
        ret_row = []
        for case in row:
            ret_row.append(case)
        ret.append(ret_row)
    return ret;

def name2cxx(name):
    return str(name).replace(".", "_")

def print_variable(cond):
    ret = "static constexpr bool VARIABLE = "
    if cond:
        ret += "true"
    else:
        ret += "false"
    print(ret + ";")

def print_parent(obj):
    if obj.parent != None:
        if obj.parent_type == "BONE":
            pname  = name2cxx(obj.parent.name) + "/"
            pname += name2cxx(obj.parent_bone)
            return name2cxx(pname)
        else:
            return name2cxx(obj.parent.name)
    else:
        return None

def print_bone_parent(obj, armature_name, default_parent = None):
    if obj.parent != None:
        return armature_name + "/" + name2cxx(obj.parent.name)
    else:
        return default_parent

def print_pose_bone_parent(obj, armature_name, default_parent = None):
    if obj.parent != None:
        return armature_name + "/" + name2cxx(obj.parent.name) + "/last"
    else:
        return default_parent

def get_links_from_bone(bone, armature_name, ret_list = []):
    ret = {}
    ret["name"] = armature_name + "/" + name2cxx(bone.name)
    #ret["is_variable"] = False
    ret["parent"] = print_bone_parent(bone, armature_name)
    #ret["matrix"] = matrix2cas(bone_matrix(bone))
    ret_list.append(ret)
    return ret_list

def get_links_from_armature(armature, ret_list = []):
    ret = {}
    ret["name"] = "armatures/" + name2cxx(armature.name)
    ret["parent"] = None
    ret_list.append(ret)
    #ret["is_variable"] = False
    #ret["bones"] = []
    for b in armature.bones:
        ret_list = get_links_from_bone(b, ret["name"], ret_list)
    return ret_list

def print_dof_matrix(name):
    ret  = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    if name == "rx":
        ret[1][1] = "cos(x)"
        ret[1][2] = "-sin(x)"
        ret[2][2] = "cos(x)"
        ret[2][1] = "sin(x)"
    elif name == "ry":
        ret[0][0] = "cos(x)"
        ret[0][2] = "-sin(x)"
        ret[2][2] = "cos(x)"
        ret[2][0] = "sin(x)"
    elif name == "rz":
        ret[0][0] = "cos(x)"
        ret[0][1] = "-sin(x)"
        ret[1][1] = "cos(x)"
        ret[1][0] = "sin(x)"
    return ret

def get_link_from_dof(name, prefix, cond, last, ret_list = []):
    if(cond):
        ret = {}
        ret["name"] = prefix + "/" + name
        ret["is_variable"] = True
        ret["parent"] = last["name"]
        ret["matrix"] = print_dof_matrix(name)
        ret_list.append(ret)
        return (ret, ret_list)
    else:
        return (last, ret_list)

def find_bone_armature(bone):
    for a in bpy.data.armatures:
        for b in a.bones:
            if b == bone:
                return (a, b)
    return (None, None)

def print_endpoint(bone):
    ret = {}
    ret["name"] = "endpoint"
    ret["is_variable"] = False
    ret["parent"] = "last"
    mat = bone.matrix_local.inverted() * bone.tail_local
    return ret

def get_links_from_pose_bone(pose_bone, armature_name, ret_list = []):
    ret = {}
    ret["name"] = armature_name + "/" + name2cxx(pose_bone.name)
    (a, b) = find_bone_armature(pose_bone.bone)
    ret["parent"] = print_pose_bone_parent(pose_bone, name2cxx(armature_name), name2cxx(armature_name))
    ret["matrix"] = matrix2cas(bone_matrix(b))
    ret_list.append(ret)
    ret["is_variable"] = False
    last = ret
    (last, ret_list) = get_link_from_dof("rx", ret["name"], not pose_bone.lock_rotation[0], last, ret_list)
    (last, ret_list) = get_link_from_dof("ry", ret["name"], not pose_bone.lock_rotation[1], last, ret_list)
    (last, ret_list) = get_link_from_dof("rz", ret["name"], not pose_bone.lock_rotation[2], last, ret_list)
    last_ret = {}
    last_ret["name"] = ret["name"] + "/last"
    last_ret["parent"] = last["name"]
    last_ret["matrix"]  = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    last_ret["is_variable"] = False
    ret_list.append(last_ret)
    endpoint = {}
    endpoint["name"] = ret["name"] + "/endpoint"
    endpoint["parent"] = last_ret["name"]
    endpoint["matrix"] = list(map(lambda x: [x], (b.matrix_local.inverted()*b.tail_local)[:]))+[[1]]
    endpoint["is_variable"] = False
    ret_list.append(endpoint)
    return ret_list

def get_links_from_object(obj, ret_list = []):
    ret = {}
    ret["name"] = name2cxx(obj.name)
    ret["is_variable"] = False
    ret["parent"] = print_parent(obj)
    ret["matrix"] = matrix2cas(obj.matrix_local)
    ret_list.append(ret)
    if obj.pose != None:
        for b in obj.pose.bones:
            ret_list = get_links_from_pose_bone(b, ret["name"], ret_list)
    return ret_list

def get_links(ret_list = []):
    for o in bpy.data.objects:
        ret_list = get_links_from_object(o, ret_list)
    return ret_list
        

def check_invalid_parent(links):
    valid = set([None])
    parents = set()
    for l in links:
        valid.add(l["name"])
    for l in links:
        parents.add(l["parent"])
    return parents - valid

def check_doubles(links):
    names = set()
    for l in links:
        if l["name"] not in names:
            names.add(l["name"])
        else:
            return True
    return False

def get_chains(links):
    ret = []
    for l in links:
        chain = []
        parent = l
        while parent != None:
            chain = [parent["name"]] + chain
            next = None
            for p in links:
                if p["name"] == parent["parent"]:
                    next = p
            parent = next
        ret.append(chain)
    return ret

def check_missing_matrix(links):
    ret = set()
    for l in links:
        if "matrix" not in l:
            ret.add(l["name"])
    return ret
            
def check_missing_is_variable(links):
    ret = set()
    for l in links:
        if "is_variable" not in l:
            ret.add(l["name"])
    return ret
            

if __name__ == '__main__':
    links = get_links()
    assert(check_invalid_parent(links) == set())
    assert(check_doubles(links) == False)
    assert(check_missing_matrix(links) == set())
    assert(check_missing_is_variable(links) == set())
    print("begin_data")
    print(json.dumps(links, indent=4))
    print("end_data")
