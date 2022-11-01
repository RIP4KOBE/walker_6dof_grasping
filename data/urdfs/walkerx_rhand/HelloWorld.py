import pybullet as p
import time
import pybullet_data
import datetime

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
gravId = p.addUserDebugParameter("gravity",-10,10,-10)  #图形化重力参数，取值-10～10,默认-10
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0.9]
startOrientation = p.getQuaternionFromEuler([0,0,0])
walkerxId = p.loadURDF("hand.urdf")
p.resetBasePositionAndOrientation(walkerxId, startPos, startOrientation) #初始化模型位姿

p.setPhysicsEngineParameter(numSolverIterations=100) #设置最大迭代次数
p.changeDynamics(walkerxId,-1,linearDamping=0, angularDamping=0)  #修改base的动力学参数

pre_joint_velocity = [0.1, 0.01, 0.001, 0.1, 0.01, 0, 0, 0, 0, 0, 0, 0]

jointIds=[]
sensorIds=[]
paramIds=[]

for j in range(p.getNumJoints(walkerxId)):
    info = p.getJointInfo(walkerxId, j)
   # print(info)
    jointName = info[1]  #关节名字
    jointType = info[2]  #关节类别
    minPosition = info[8] #关节最小限位
    maxPosition = info[9] #关节最大限位
    if (jointType == p.JOINT_REVOLUTE):
        jointIds.append(j)
        paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), minPosition, maxPosition, 0.0))
        p.resetJointState(walkerxId, j, 0.0)
    if (jointType == p.JOINT_FIXED):
        sensorIds.append(j)
        p.enableJointForceTorqueSensor(walkerxId, j)

p.setRealTimeSimulation(0)
p.setTimeStep(0.001)
starttime = datetime.datetime.now()
count = 0
p.getCameraImage(320, 200)
p.setGravity(0, 0, 0)

# test
# joint_list = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
# joint_state = p.getJointStates(walkerxId, joint_list)
# joint_position = [x[0] for x in joint_state]
# joint_velocity = [x[1] for x in joint_state]
# joint_acc = [(y-x)/0.001 for x,y in zip(joint_velocity,pre_joint_velocity)]
# print(joint_position)
# print(joint_velocity)
# print(joint_acc)

while 1:
    # p.getCameraImage(320, 200)
    # p.setGravity(0, 0, p.readUserDebugParameter(gravId))
    p.setGravity(0, 0, 0)
    for i in range(len(paramIds)):
        c = paramIds[i]
        targetPos = p.readUserDebugParameter(c)
        # targetPos = 0.0
        p.setJointMotorControl2(walkerxId, jointIds[i], p.POSITION_CONTROL, targetPos)
    # time.sleep(0.01)
    #leftInfo = p.getJointState(walkerxId, sensorIds[0])
    #rightInfo = p.getJointState(walkerxId, sensorIds[1])
    #count += 1
    # print(count)
    #if (count == 10000):
    #    endtime = datetime.datetime.now()
    #    print((endtime - starttime).seconds)
    # print(leftInfo[2])
    # print(rightInfo[2])
    # left_6axis = p.getJointState(walkerxId, 7)[2]
    # right_6axis = p.getJointState(walkerxId, 14)[2]
    # print(left_6axis)
    # print(right_6axis)
    # IMU = p.getLinkState(walkerxId, 15)
    # print(IMU)

    p.stepSimulation()

p.disconnect()
