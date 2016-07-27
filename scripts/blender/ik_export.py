import bpy

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
    vals = []
    for row in mat:
        for case in row:
            vals += [real2cas(case)]
    ret  = "CAS::General<double, double>::"
    ret += "Matrix<4,4>::"
    ret += "Any<"
    ret += ','.join(vals)
    ret += ">"
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
            print("using parent = " + name2cxx(pname) + ";")
        else:
            print("using parent = " + name2cxx(obj.parent.name) + ";")
    else:
        print("using parent = void;")

def print_bone_parent(obj, default_parent = 'void'):
    if obj.parent != None:
        print("using parent = " + name2cxx(obj.parent.name) + ";")
    else:
        print("using parent = "+ default_parent +";")

def print_pose_bone_parent(obj, default_parent = 'void'):
    if obj.parent != None:
        print("using parent = " + name2cxx(obj.parent.name) + "::last;")
    else:
        print("using parent = "+ default_parent +";")

def print_bone(bone):
    print("struct " + name2cxx(bone.name) + " {")
    print_variable(False)
    print_bone_parent(bone)
    mat  = "using matrix = "
    mat += matrix2cas(bone_matrix(bone)) + ";"
    print(mat)
    print("};")

def print_armature(armature):
    print("struct " + name2cxx(armature.name) + " {")
    print_variable(False)
    for b in armature.bones:
        print_bone(b)
    print("};")

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
        print("struct " + name + " {")
        print_variable(True)
        print("using parent = " + last + ";")
        print_dof_matrix(name)
        print("};")
        return name;
    return last

def find_bone_armature(bone):
    for a in bpy.data.armatures:
        for b in a.bones:
            if b == bone:
                return a
    return None

def print_endpoint(bone):
    print("struct endpoint {")
    print_variable(False)
    print("using parent = last;")
    tloc = bone.matrix_local.inverted() * bone.tail_local
    ret  = "using matrix = "
    ret += "CAS::General<double, double>::"
    ret += "Space3D::"
    ret += "Translation<"
    ret += real2cas(tloc[0]) + ","
    ret += real2cas(tloc[1]) + ","
    ret += real2cas(tloc[2])
    ret += ">;"
    print(ret)
    print("};")


def print_pose_bone(pose_bone, armature_name):
    struct_def = "struct " + name2cxx(pose_bone.name)
    a = find_bone_armature(pose_bone.bone)
    if a != None:
        struct_def += " : public armatures::"
        struct_def += name2cxx(a.name) + "::"
        struct_def += name2cxx(pose_bone.bone.name)
    print(struct_def + " {")    
    print_variable(False)
    print_pose_bone_parent(pose_bone, name2cxx(armature_name))
    last = name2cxx(pose_bone.name)
    last = print_dof("rx", not pose_bone.lock_rotation[0], last)
    last = print_dof("ry", not pose_bone.lock_rotation[1], last)
    last = print_dof("rz", not pose_bone.lock_rotation[2], last)
    print("using last = " + last + ";")
    print_endpoint(pose_bone.bone)
    print("};")

def print_object(obj):
    print("struct " + name2cxx(obj.name) + " {")
    print_variable(False)
    print_parent(obj)
    mat  = "using matrix = "
    mat += matrix2cas(obj.matrix_local) + ";"
    print(mat)
    if obj.pose != None:
        print("struct bones {")
        for b in obj.pose.bones:
            print_pose_bone(b, obj.name)
        print("};")
    print("};")

def print_obj_tree():
    visited = {}
    while len(visited) != len(bpy.data.objects):
        for o in bpy.data.objects:
            if not o.name in visited:
                if (o.parent == None) or (o.parent.name in visited):
                    visited[o.name] = o
                    print_object(o)
        

print("namespace ik_export {")

print("namespace armatures {")
for a in bpy.data.armatures:
    print_armature(a)
print("}")
    
print("namespace objects {")
print_obj_tree()
print("}")

print("}")
