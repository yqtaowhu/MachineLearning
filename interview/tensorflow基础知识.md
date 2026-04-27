- [tensorflow基础知识](#tensorflow%e5%9f%ba%e7%a1%80%e7%9f%a5%e8%af%86)
  - [1. name scope和variable scope区别](#1-name-scope%e5%92%8cvariable-scope%e5%8c%ba%e5%88%ab)
  - [参考资料](#%e5%8f%82%e8%80%83%e8%b5%84%e6%96%99)


# tensorflow基础知识

## 1. name scope和variable scope区别

**name_scope 和 variable_scope 主要是因为 变量共享 的需求**

1. 对于使用tf.Variable来说，tf.name_scope和tf.variable_scope功能一样，都是给变量加前缀，相当于分类管理，模块化。
2. 对于tf.get_variable来说，tf.name_scope对其无效，也就是说tf认为当你使用tf.get_variable时，你只归属于tf.variable_scope来管理共享与否。




## 参考资料
- [tensorflow里面name_scope, variable_scope等如何理解？](https://www.zhihu.com/question/54513728/answer/181819324)