from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal

from comm_demo.core.session import SimulationSession

"""定义一个名为 StepWorker 的类，继承自 QObject，用于在单独的线程中执行仿真步骤，并通过信号与主线程通信"""
class StepWorker(QObject):
    finished = pyqtSignal(str, object) # 发射完成信号，包含步骤说明和结果
    failed = pyqtSignal(str) # 发射失败信号，包含错误信息

    def __init__(self, session: SimulationSession):
        super().__init__()
        self.session = session

    def run(self):
        try:
            note = self.session.step()
            result = self.session.build_result() if self.session.is_finished() else None
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(note, result)
