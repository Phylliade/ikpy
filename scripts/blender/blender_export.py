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
            pname  = "armatures::"
            pname += name2cxx(obj.parent.name) + "::"
            pname += name2cxx(obj.parent_bone)
            return name2cxx(pname)
        else:
            return name2cxx(obj.parent.name)
    else:
        return None

def print_bone_parent(obj, default_parent = 'void'):
    if obj.parent != None:
        return name2cxx(obj.parent.name)
    else:
        return default_parent

def print_pose_bone_parent(obj, default_parent = 'void'):
    if obj.parent != None:
        return name2cxx(obj.parent.name) + ".last"
    else:
        return default_parent

def print_bone(bone):
    ret = {}
    ret["name"] = name2cxx(bone.name)
    #ret["is_variable"] = False
    ret["parent"] = print_bone_parent(bone)
    #ret["matrix"] = matrix2cas(bone_matrix(bone))
    return ret

def print_armature(armature):
    ret = {}
    ret["name"] = "[armature]" + name2cxx(armature.name)
    #ret["is_variable"] = False
    ret["bones"] = []
    for b in armature.bones:
        ret["bones"].append(print_bone(b))
    return ret

def print_dof_matrix(name):
    ret  = "using matrix = "
    ret += "CAS::General<double, double>::"
    ret += "Space3D::"
    if name == "rx":
        ret += "RotationX"
    elif name == "ry":
        ret += "RotationY"
    elif name == "rz":
        ret += "RotationZ"
    ret += ";"
    print(ret)

def print_dof(name, cond, last):
    if(cond):
        ret = {}
        ret["name"] = name
        #ret["is_variable"] = True
        ret["parent"] = last
        #ret["matrix"] = print_dof_matrix(name)
        return (ret, name)
    else:
        return (None, last)

def find_bone_armature(bone):
    for a in bpy.data.armatures:
        for b in a.bones:
            if b == bone:
                return a
    return None

def print_endpoint(bone):
    ret = {}
    ret["name"] = "endpoint"
    #ret["is_variable"] = False
    ret["parent"] = "last"
    mat = bone.matrix_local.inverted() * bone.tail_local
    return ret

def print_pose_bone(pose_bone, armature_name):
    ret = {}
    ret["name"] = "pose_bones/" + name2cxx(pose_bone.name)
    a = find_bone_armature(pose_bone.bone)
    if a != None:
        ret["armature"] = name2cxx(a.name)
        ret["armature_bone"] = name2cxx(pose_bone.bone.name)
    #ret["is_variable"] = False
    ret["parent"] = print_pose_bone_parent(pose_bone, name2cxx(armature_name))
    last = name2cxx(pose_bone.name)
    (ret["rx"], last) = print_dof("rx", not pose_bone.lock_rotation[0], last)
    (ret["ry"], last) = print_dof("ry", not pose_bone.lock_rotation[1], last)
    (ret["rz"], last) = print_dof("rz", not pose_bone.lock_rotation[2], last)
    ret["last"] = last
    ret["endpoint"] = print_endpoint(pose_bone.bone)
    return ret

def print_object(obj):
    ret = {}
    ret["name"] = "objects/" + name2cxx(obj.name)
    #ret["is_variable"] = False
    ret["parent"] = print_parent(obj)
    #ret["matrix"] = matrix2cas(obj.matrix_local)
    if obj.pose != None:
        ret["bones"] = []
        for b in obj.pose.bones:
            ret["bones"].append(print_pose_bone(b, obj.name))
    return ret

def get_obj_tree():
    visited = {}
    while len(visited) != len(bpy.data.objects):
        for o in bpy.data.objects:
            if not o.name in visited:
                if (o.parent == None) or (o.parent.name in visited):
                    visited[o.name] = print_object(o)
    return visited
        

if __name__ == '__main__':
    for a in bpy.data.armatures:
        print_armature(a)
    print(json.dumps(get_obj_tree(), indent=4))
