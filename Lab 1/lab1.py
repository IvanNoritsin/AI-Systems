import numpy as np
import matplotlib.pyplot as plt

data_path = "glass.data"
data = np.genfromtxt(data_path, delimiter=",")

Rl = []
Na = []
Mg = []
Al = []
Si = []
K = []
Ca = []
Ba = []
Fe = []

for dot in data:
    Rl.append(dot[1])
    Na.append(dot[2])
    Mg.append(dot[3])
    Al.append(dot[4])
    Si.append(dot[5])
    K.append(dot[6])
    Ca.append(dot[7])
    Ba.append(dot[8])
    Fe.append(dot[9])
    
print("Cреднее значение Rl:", np.mean(Rl))
print("Cреднее значение Na:", np.mean(Na))
print("Cреднее значение Mg:", np.mean(Mg))
print("Cреднее значение Al:", np.mean(Al))
print("Cреднее значение Si:", np.mean(Si))
print("Cреднее значение K:", np.mean(K))
print("Cреднее значение Ca:", np.mean(Ca))
print("Cреднее значение Ba:", np.mean(Ba))
print("Cреднее значение Fe:", np.mean(Fe))
print(" ")
print("Максимальное значение Rl:", np.max(Rl))
print("Максимальное значение Na:", np.max(Na))
print("Максимальное значение Mg:", np.max(Mg))
print("Максимальное значение Al:", np.max(Al))
print("Максимальное значение Si:", np.max(Si))
print("Максимальное значение K:", np.max(K))
print("Максимальное значение Ca:", np.max(Ca))
print("Максимальное значение Ba:", np.max(Ba))
print("Максимальное значение Fe:", np.max(Fe))
print(" ")
print("Минимальное значение Rl:", np.min(Rl))
print("Минимальное значение Na:", np.min(Na))
print("Минимальное значение Mg:", np.min(Mg))
print("Минимальное значение Al:", np.min(Al))
print("Минимальное значение Si:", np.min(Si))
print("Минимальное значение K:", np.min(K))
print("Минимальное значение Ca:", np.min(Ca))
print("Минимальное значение Ba:", np.min(Ba))
print("Минимальное значение Fe:", np.min(Fe))

    
plt.figure(1)
plt.plot(Na[:70], Mg[:70], 'ro', label='Building windows (float processed)')
plt.plot(Na[70:146], Mg[70:146], 'bs', label='Building windows (non float processed)')
plt.plot(Na[146:163], Mg[146:163], 'gd', label='Vehicle windows (float processed)')
plt.plot(Na[163:176], Mg[163:176], 'm^', label='Containers')
plt.plot(Na[176:185], Mg[176:185], 'yp', label='Tableware')
plt.plot(Na[185:214], Mg[185:214], 'c*', label='Headlamps')
plt.legend()
plt.xlabel('Натрий (Na)')
plt.ylabel('Магний (Mg)')

plt.figure(2)
plt.plot(Al[:70], Si[:70], 'ro', label='Building windows (float processed)')
plt.plot(Al[70:146], Si[70:146], 'bs', label='Building windows (non float processed)')
plt.plot(Al[146:163], Si[146:163], 'gd', label='Vehicle windows (float processed)')
plt.plot(Al[163:176], Si[163:176], 'm^', label='Containers')
plt.plot(Al[176:185], Si[176:185], 'yp', label='Tableware')
plt.plot(Al[185:214], Si[185:214], 'c*', label='Headlamps')
plt.legend()
plt.xlabel('Алюминий (Al)')
plt.ylabel('Кремний (Si)')

plt.figure(3)
plt.plot(Ca[:70], Ba[:70], 'ro', label='Building windows (float processed)')
plt.plot(Ca[70:146], Ba[70:146], 'bs', label='Building windows (non float processed)')
plt.plot(Ca[146:163], Ba[146:163], 'gd', label='Vehicle windows (float processed)')
plt.plot(Ca[163:176], Ba[163:176], 'm^', label='Containers')
plt.plot(Ca[176:185], Ba[176:185], 'yp', label='Tableware')
plt.plot(Ca[185:214], Ba[185:214], 'c*', label='Headlamps')
plt.legend()
plt.xlabel('Кальций (Ca)')
plt.ylabel('Барий (Ba)')

plt.figure(4)
plt.plot(K[:70], Fe[:70], 'ro', label='Building windows (float processed)')
plt.plot(K[70:146], Fe[70:146], 'bs', label='Building windows (non float processed)')
plt.plot(K[146:163], Fe[146:163], 'gd', label='Vehicle windows (float processed)')
plt.plot(K[163:176], Fe[163:176], 'm^', label='Containers')
plt.plot(K[176:185], Fe[176:185], 'yp', label='Tableware')
plt.plot(K[185:214], Fe[185:214], 'c*', label='Headlamps')
plt.legend()
plt.xlabel('Калий (K)')
plt.ylabel('Железо (Fe)')

plt.figure(5)
plt.plot(Na[:70], Ca[:70], 'ro', label='Building windows (float processed)')
plt.plot(Na[70:146], Ca[70:146], 'bs', label='Building windows (non float processed)')
plt.plot(Na[146:163], Ca[146:163], 'gd', label='Vehicle windows (float processed)')
plt.plot(Na[163:176], Ca[163:176], 'm^', label='Containers')
plt.plot(Na[176:185], Ca[176:185], 'yp', label='Tableware')
plt.plot(Na[185:214], Ca[185:214], 'c*', label='Headlamps')
plt.legend()
plt.xlabel('Натрий (Na)')
plt.ylabel('Кальций (Ca)')

