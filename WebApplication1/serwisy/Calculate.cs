namespace WebApplication1.serwisy
{
    public class Calculate
    {
        public Calculate(int initVal)
        {
            Value = initVal;
        }

        private int Value { get; set; }

        public void Add(int value)
        {
            this.Value += value;
        }

        public void Decrease(int value)
        {
            this.Value -= value;
        }

        public int GetValue()
        {
            return Value;
        }
    } 
}
