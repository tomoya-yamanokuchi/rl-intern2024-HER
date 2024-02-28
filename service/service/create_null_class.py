


def create_null_class(base_class):
    class_name = f"Null{base_class.__name__}"

    def null_method(*args, **kwargs):
        pass

    methods = {name: null_method for name, _ in base_class.__dict__.items() if callable(getattr(base_class, name))}
    return type(class_name, (base_class,), methods)


class MyService:
    def foo(self):
        print("Foo called")

    def bar(self):
        print("Bar called")


if __name__ == '__main__':
    NullMyService = create_null_class(MyService)
    null_instance = NullMyService()
    null_instance.foo()  # 何も出力されない
    null_instance.bar()  # 何も出力されない
