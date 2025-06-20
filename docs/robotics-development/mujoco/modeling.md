# 建模

## 简介

MuJoCo 可以加载其原生 MJCF 格式的 XML 模型文件，也支持更为流行但功能相对有限的 URDF 格式。本章节是 MJCF 建模指南。详细的参考手册可在"XML 参考"章节中查阅。关于 URDF 的文档可在其他地方找到；在此我们仅描述 MuJoCo 特有的 URDF 扩展。

MJCF 模型能够表示具有广泛特性和模型元素的复杂动力学系统。要访问所有这些功能，需要一种丰富的建模格式，如果在设计时不考虑易用性，这种格式可能会变得繁琐。因此，我们努力将 MJCF 设计为一种可扩展的格式，使用户可以从简单开始，随后构建更详细的模型。特别有帮助的是受到 HTML 内联 CSS（级联样式表）启发的广泛默认设置机制。这使用户能够快速创建新模型并进行实验。众多可用于重新配置仿真管道的选项，以及快速重新加载功能，进一步辅助了实验过程，使模型编辑成为一个交互式过程。

MJCF 可以被视为建模格式和编程语言的混合体。它内置了编译器，这通常是与编程语言相关的概念。虽然 MJCF 不具备通用编程语言的全部功能，但根据模型的设计方式，系统会自动调用许多复杂的编译时计算。

### 加载模型

正如"概述"章节中"模型实例"部分所解释的，MuJoCo 模型可以从 MJCF 或 URDF 格式的纯文本 XML 文件中加载，然后编译成底层的 mjModel。另外，之前保存的 mjModel 也可以直接从二进制 MJB 文件加载——虽然该文件格式未公开文档，但本质上是 mjModel 内存缓冲区的一个副本。MJCF 和 URDF 文件通过 [ `mj_loadXML` ](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-loadxml) 函数加载，而 MJB 文件则通过 [ `mj_loadModel` ](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-loadmodel) 函数加载。

当加载 XML 文件时，系统首先使用内部的 TinyXML 解析器将其解析为文档对象模型（DOM）。然后，这个 DOM 被处理并转换为高级 [ `mjSpec` ](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjspec) 对象。这种转换取决于模型格式——格式是从 XML 文件的顶层元素推断出来的，而非从文件扩展名判断。需要注意的是，有效的 XML 文件必须有唯一的顶层元素。对于 MJCF 格式，这个元素必须是 mujoco；对于 URDF 格式，则必须是 robot。

### 编译模型

一旦创建了高级 [ `mjSpec` ](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjspec)  对象——无论是通过加载 MJCF 文件、URDF 文件，还是以编程方式创建——它就会被编译成 [ `mjModel` ](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjmodel)。编译过程独立于加载过程，这意味着无论 [ `mjSpec` ](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjspec)  是如何创建的，编译器的工作方式都是一样的。解析器和编译器都会进行全面的错误检查，并在遇到第一个错误时中止。产生的错误信息包含 XML 文件中的行号和列号，这些信息通常都很明确，因此我们在此不再赘述。解析器使用自定义模式来确保文件结构、元素和属性的有效性。随后，编译器会应用许多额外的语义检查。最后，系统会执行一步编译后模型的仿真，并拦截任何运行时错误。这是通过（临时）将 [ `mju_user_error` ](https://mujoco.readthedocs.io/en/stable/APIreference/APIglobals.html#mju-user-error) 设置为指向一个抛出 C++ 异常的函数来实现的；如有需要，用户可以在运行时实现类似的错误拦截功能。

整个解析和编译过程非常快速——如果模型不包含大型网格或需要通过仿真计算的执行器长度范围，整个过程通常不到一秒。这使得用户可以通过频繁重新加载和可视化更改来交互式地设计模型。值得注意的是，[ `simulate.cc` ](https://mujoco.readthedocs.io/en/stable/programming/samples.html#sasimulate) 代码示例中有一个重新加载当前模型的键盘快捷键（Ctrl+L）。

### 保存模型

一个 MJCF 模型可以由多个（包含的）XML 文件以及 XML 中引用的网格、高度场和纹理组成。编译后，所有这些文件的内容都会被整合到 mjModel 中，可以通过 [ `mj_saveModel` ](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-savemodel) 函数将其保存为二进制 MJB 文件。MJB 是一个独立的文件，不引用任何其他文件，并且加载速度更快。因此，我们建议将常用模型保存为 MJB 格式，并在需要进行仿真时加载它们。

还可以使用 [ `mj_saveLastXML` ](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-savelastxml) 函数将编译后的 [ `mjSpec` ](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjspec)  保存为 MJCF 格式。如果对应的 mjModel 中的任何实值字段在编译后被修改（这种情况不常见，但在系统识别应用中可能发生），这些修改会在保存前自动复制回 [ `mjSpec` ](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjspec) 。需要注意的是，无法在编译后的模型中进行结构性更改。XML 写入器会尝试生成最小的 MJCF 文件，该文件保证能够编译成相同的模型，只有因实值的纯文本表示而导致的微小数值差异。生成的文件结构可能与原始文件不同，因为 MJCF 有许多用户便利功能，允许以不同方式指定相同的模型。XML 写入器使用 MJCF 的"规范"子集，其中所有坐标都是局部的，所有刚体位置、方向和惯性属性都被明确指定。在"计算"章节中，我们展示了一个 MJCF 文件示例及其对应的保存结果。

### 编辑模型

从 MuJoCo 3.2 版本开始，可以使用 [ `mjSpec` ](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjspec)  结构体和相关 API 创建和修改模型。更多详细信息，请参阅"模型编辑"章节。

## 模型机理

MJCF 在模型创建过程中使用了几种跨越多个模型元素的机制。为避免重复，我们将在本节中详细描述这些机制。这些机制并不对应"计算"章节中介绍的新模拟概念，而是旨在简化 MJCF 模型的创建，并使用户能够使用不同的数据格式，而无需手动转换为标准格式。

### 运动学树(Kinematic Tree)

MJCF 文件的主要部分是由嵌套的 body 元素创建的 XML 树。顶层 body 比较特殊，称为 worldbody（世界刚体）。这种树状组织结构与 URDF 形成对比，在 URDF 中，用户创建一组链接，然后通过指定子链接和父链接的关节来连接它们。而在 MJCF 中，子刚体在 XML 的意义上就是父刚体的子元素。

当在刚体内定义关节时，其功能并非连接父子刚体，而是在它们之间创建运动自由度。如果某个刚体内没有定义关节，则该刚体会被焊接到其父刚体上。MJCF 中的一个刚体可以包含多个关节，因此不需要为创建复合关节而引入虚拟刚体。相反，只需在同一刚体内定义所有构成所需复合关节的基本关节即可。例如，两个滑块和一个铰链可以用来建模在平面内移动的刚体。

其他 MJCF 元素可以在由嵌套刚体元素创建的树中定义，特别是 joint（关节）、geom（几何体）、site（位点）、camera（相机）和 light（光源）。当元素在刚体内定义时，它就固定在该刚体的局部坐标系中，并始终随之移动。引用多个刚体或根本不引用刚体的元素则在运动学树外的独立部分中定义。

### 默认设置

MJCF 拥有一套完善的默认属性值设置机制。这使我们能够拥有大量元素和属性来展示软件的丰富功能，同时编写简短易读的模型文件。这种机制还使用户能够在一处进行更改，并使其在整个模型中传播。我们先来看一个例子。

```xml
<mujoco>
  <default class="main">
    <geom rgba="1 0 0 1"/>
    <default class="sub">
      <geom rgba="0 1 0 1"/>
    </default>
  </default>

  <worldbody>
    <geom type="box"/>
    <body childclass="sub">
      <geom type="ellipsoid"/>
      <geom type="sphere" rgba="0 0 1 1"/>
      <geom type="cylinder" class="main"/>
    </body>
  </worldbody>
</mujoco>
```

这个例子实际上无法编译，因为缺少一些必要信息，但在这里我们只关注几何体（geom）rgba 值的设置。由于默认设置机制，上面创建的四个几何体最终会具有以下 rgba 值：

|几何体类型|集合体 rgba|
|---|---|
|box|1 0 0 1|
|elipsoid|0 1 0 1|
|sphere|0 0 1 1|
|cylinder|1 0 0 1|

盒子（box）使用顶层默认类"main"来设置其未定义的属性，因为没有指定其他类。而 body 元素指定了 childclass 为"sub"，这导致该刚体的所有子元素（及其子元素的子元素等）都使用"sub"类，除非另有指定。因此椭球体（ellipsoid）使用"sub"类。球体（sphere）显式定义了 rgba 值，这会覆盖默认设置。圆柱体（cylinder）指定了默认类为"main"，所以它使用"main"而非"sub"，尽管后者是在包含该几何体的刚体的 childclass 属性中指定的。

现在我们来描述一般规则。MuJoCo 支持无限数量的默认类，这些类由 XML 中可能嵌套的 default 元素创建。每个类都有一个唯一的名称——这是必需的属性，除了顶层类，如果未定义，其名称为"main"。每个类还包含一整套虚拟模型元素，其属性设置如下。当一个默认类在另一个默认类内定义时，子类会自动继承父类的所有属性值。然后它可以通过定义相应的属性来覆盖部分或全部属性值。顶层默认类没有父类，因此其属性被初始化为内部默认值，这些值在"参考"章节中有所展示。

默认类中包含的虚拟元素不是模型的一部分；它们仅用于初始化实际模型元素的属性值。当首次创建一个实际元素时，其所有属性都从当前活动默认类中对应的虚拟元素复制而来。系统总是有一个活动的默认类，可以通过以下三种方式之一确定：如果当前元素或其任何祖先刚体都未指定类，则使用顶层类（无论它是叫"main"还是其他名称）。如果当前元素未指定类，但其一个或多个祖先刚体指定了 childclass，则使用最近祖先刚体的 childclass。如果当前元素指定了一个类，则使用该类，无论其祖先刚体中的任何 childclass 属性如何。

某些属性，如刚体惯性，可以处于特殊的未定义状态。这指示编译器从其他信息中推断相应的值，在这种情况下是从附加到刚体的几何体的惯性中推断。未定义状态无法在 XML 文件中输入。因此，一旦某个属性在给定类中被定义，它就不能在该类或其任何子类中被取消定义。所以，如果目标是在给定模型元素中保持某个属性未定义，那么它必须在活动默认类中也是未定义的。

最后一个特殊点是执行器（actuators）。它们不同是因为一些与执行器相关的元素实际上是快捷方式，而快捷方式与默认设置机制的交互方式并不那么直观。这将在下面的"执行器快捷方式"部分中解释。
