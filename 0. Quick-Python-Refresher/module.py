# 1. No Args , No Return type
def func1():
    var1 = int(input("Var1:"))
    var2 = int(input("Var2:"))
    var3 = var1 + var2
    print("Addition:",var3)

# 2. With Args , No Return type
def func2(var1, var2):
    var3 = var1 + var2
    print("Addition:",var3)

# 3. No Args , with Return type
def func3():
    var1 = int(input("Var1:"))
    var2 = int(input("Var2:"))
    var3 = var1 + var2
    return var3

# 4. With Args , With Return type
def func4(var1, var2):
    var3 = var1 + var2
    return var3